"""Phase 1 utility and systems evaluation."""

import math
import time
from collections import Counter

import torch
import torch.nn.functional as F

from spidlu.metrics import cuda_memory


def _labels_from_batch(batch, device):
    input_ids = batch["input_ids"].to(device)
    labels = batch.get("labels", batch["input_ids"]).to(device)
    return input_ids, labels


def evaluate_nll(model, dataloader, device, max_batches=None):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    start = time.perf_counter()
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if max_batches is not None and idx >= max_batches:
                break
            input_ids, labels = _labels_from_batch(batch, device)
            outputs = model(input_ids=input_ids, labels=labels)
            logits = outputs.logits
            nll = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="sum",
            )
            total_nll += nll.item()
            total_tokens += labels.numel()
    elapsed = time.perf_counter() - start
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "token_weighted_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "eval_tokens": total_tokens,
        "eval_time": elapsed,
    }


def downstream_accuracy(model, dataloader, device, max_batches=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            if max_batches is not None and idx >= max_batches:
                break
            input_ids, labels = _labels_from_batch(batch, device)
            logits = model(input_ids=input_ids).logits
            preds = logits[:, :-1].argmax(dim=-1)
            targets = labels[:, 1:]
            correct += (preds == targets).sum().item()
            total += targets.numel()
    return {"downstream_accuracy": correct / max(total, 1), "downstream_tokens": total}


def generation_outputs(model, tokenizer, prompts, device, max_new_tokens):
    model.eval()
    outputs = []
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt").to(device)
            generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
            text = tokenizer.decode(generated[0], skip_special_tokens=True)
            outputs.append({"prompt": prompt, "output": text})
    return outputs


def repetition_diversity_stats(generations):
    tokens = []
    repeated_bigrams = 0
    total_bigrams = 0
    for item in generations:
        words = item["output"].split()
        tokens.extend(words)
        bigrams = list(zip(words, words[1:]))
        total_bigrams += len(bigrams)
        counts = Counter(bigrams)
        repeated_bigrams += sum(max(0, count - 1) for count in counts.values())
    unique = len(set(tokens))
    total = len(tokens)
    return {
        "distinct_1": unique / max(total, 1),
        "repetition_bigram_rate": repeated_bigrams / max(total_bigrams, 1),
    }


def latency_profile(model, tokenizer, prompt, device, max_new_tokens):
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    with torch.no_grad():
        start = time.perf_counter()
        outputs = model(**encoded, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        prefill_time = time.perf_counter() - start
        next_token = outputs.logits[:, -1:].argmax(dim=-1)
        past_key_values = outputs.past_key_values
        decode_start = time.perf_counter()
        decoded = 0
        for _ in range(max_new_tokens):
            out = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
            next_token = out.logits[:, -1:].argmax(dim=-1)
            past_key_values = out.past_key_values
            decoded += next_token.numel()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decode_time = time.perf_counter() - decode_start
    memory = cuda_memory()
    prefill_tokens = encoded["input_ids"].numel()
    return {
        "prefill_latency": prefill_time,
        "decode_latency_per_token": decode_time / max(decoded, 1),
        "prefill_throughput": prefill_tokens / max(prefill_time, 1e-9),
        "decode_throughput": decoded / max(decode_time, 1e-9),
        **memory,
    }
