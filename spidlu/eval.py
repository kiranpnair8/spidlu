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
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, labels, attention_mask


def shifted_causal_targets(labels, attention_mask=None):
    """Return next-token labels and a validity mask for causal-LM evaluation."""
    targets = labels[:, 1:].clone()
    valid = targets.ne(-100)
    if attention_mask is not None:
        valid &= attention_mask[:, 1:].bool()
    targets = targets.masked_fill(~valid, -100)
    return targets, valid


def causal_lm_nll_from_logits(logits, labels, attention_mask=None):
    """Sum next-token NLL over valid target tokens."""
    shift_logits = logits[:, :-1, :].contiguous()
    targets, valid = shifted_causal_targets(labels, attention_mask=attention_mask)
    nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        targets.contiguous().view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return nll, valid.sum()


def compare_hf_loss(model, input_ids, labels, attention_mask=None):
    """Compare custom shifted loss with model(..., labels=...).loss on one batch."""
    kwargs = {"input_ids": input_ids, "labels": labels}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    outputs = model(**kwargs)
    custom_nll, valid_tokens = causal_lm_nll_from_logits(
        outputs.logits,
        labels,
        attention_mask=attention_mask,
    )
    custom_loss = custom_nll / valid_tokens.clamp_min(1)
    return {
        "custom_loss": custom_loss,
        "hf_loss": outputs.loss,
        "valid_tokens": valid_tokens,
        "logits": outputs.logits,
    }


def evaluate_nll(model, dataloader, device, max_batches=None):
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for idx, batch in enumerate(dataloader):
            if max_batches is not None and idx >= max_batches:
                break
            input_ids, labels, attention_mask = _labels_from_batch(batch, device)
            kwargs = {"input_ids": input_ids}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            outputs = model(**kwargs)
            nll, valid_tokens = causal_lm_nll_from_logits(
                outputs.logits,
                labels,
                attention_mask=attention_mask,
            )
            total_nll += nll.item()
            total_tokens += valid_tokens.item()
    elapsed = time.perf_counter() - start
    mean_nll = total_nll / max(total_tokens, 1)
    return {
        "token_weighted_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "eval_tokens": total_tokens,
        "eval_time": elapsed,
    }


def downstream_accuracy(model, dataloader, device, max_batches=None):
    """Measure next-token token accuracy on the LM evaluation stream.

    This is not a task-level downstream benchmark. Chance accuracy is 1/vocab
    under a uniform-token baseline, and exact sequence accuracy is reported
    separately as a stricter per-example diagnostic.
    """
    model.eval()
    correct = 0
    total = 0
    exact_correct = 0
    exact_total = 0
    vocab_size = None
    examples = []
    with torch.inference_mode():
        for idx, batch in enumerate(dataloader):
            if max_batches is not None and idx >= max_batches:
                break
            input_ids, labels, attention_mask = _labels_from_batch(batch, device)
            kwargs = {"input_ids": input_ids}
            if attention_mask is not None:
                kwargs["attention_mask"] = attention_mask
            logits = model(**kwargs).logits
            vocab_size = logits.size(-1)
            preds = logits[:, :-1].argmax(dim=-1)
            targets, valid = shifted_causal_targets(labels, attention_mask=attention_mask)
            matches = preds.eq(targets) & valid
            correct += matches.sum().item()
            total += valid.sum().item()
            per_example_valid = valid.sum(dim=1) > 0
            per_example_exact = (matches | ~valid).all(dim=1) & per_example_valid
            exact_correct += per_example_exact.sum().item()
            exact_total += per_example_valid.sum().item()
            if len(examples) < 4:
                row = 0
                valid_positions = valid[row].nonzero(as_tuple=False).flatten()
                take = valid_positions[:8]
                examples.append({
                    "predictions": preds[row, take].detach().cpu().tolist(),
                    "labels": targets[row, take].detach().cpu().tolist(),
                    "matches": matches[row, take].detach().cpu().tolist(),
                })
    chance = (1.0 / vocab_size) if vocab_size else None
    return {
        "downstream_accuracy": correct / max(total, 1),
        "downstream_metric": "next_token_token_accuracy",
        "downstream_chance_accuracy": chance,
        "downstream_sequence_exact_accuracy": exact_correct / max(exact_total, 1),
        "downstream_tokens": total,
        "downstream_examples": examples,
    }


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
