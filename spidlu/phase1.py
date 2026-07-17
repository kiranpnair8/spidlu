"""End-to-end Phase 1 RQ1 runner."""

import hashlib
import json
from pathlib import Path

import torch

from spidlu.data import load_tokenizer, make_dataloader, make_lm_datasets
from spidlu.eval import (
    downstream_accuracy,
    evaluate_nll,
    generation_outputs,
    latency_profile,
    repetition_diversity_stats,
)
from spidlu.metrics import count_parameters, environment_metadata, relative_perplexity_change
from spidlu.seed import set_seed
from spidlu.surgery import Variant, apply_activation_surgery
from spidlu.train import train_variant


TRAINED_VARIANTS = {
    Variant.SPIDLU,
    Variant.ANN_COMPUTE_MATCHED,
    Variant.QUANTIZED_ACTIVATION,
}


def load_causal_lm(cfg):
    from transformers import AutoModelForCausalLM

    kwargs = {}
    if cfg.model_revision:
        kwargs["revision"] = cfg.model_revision
    return AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **kwargs)


def state_fingerprint(model, tensors=8):
    h = hashlib.sha256()
    count = 0
    for _, tensor in model.state_dict().items():
        h.update(tensor.detach().cpu().float().contiguous().numpy().tobytes())
        count += 1
        if count >= tensors:
            break
    return h.hexdigest()


def build_variant_model(cfg, variant, device):
    set_seed(cfg.seed)
    model = load_causal_lm(cfg)
    base_fingerprint = state_fingerprint(model)
    records = apply_activation_surgery(model, variant, cfg)
    model.to(device)
    return model, base_fingerprint, records


def run_variant(cfg, variant, tokenizer, datasets, baseline_perplexity=None):
    variant = Variant(variant)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, base_fingerprint, replacements = build_variant_model(cfg, variant, device)

    train_loader = make_dataloader(
        datasets[cfg.train_split],
        batch_size=cfg.batch_size,
        shuffle=True,
        seed=cfg.seed,
    )
    eval_loader = make_dataloader(
        datasets[cfg.eval_split],
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        seed=cfg.seed,
    )
    downstream_loader = make_dataloader(
        datasets[cfg.downstream_split],
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        seed=cfg.seed,
    )

    result = {
        "variant": variant.value,
        "base_weight_fingerprint": base_fingerprint,
        "replacement_records": [record.__dict__ for record in replacements],
        **count_parameters(model),
    }

    if variant in TRAINED_VARIANTS:
        result.update(train_variant(model, train_loader, cfg, device))
    else:
        result.update({
            "processed_tokens": 0,
            "optimizer_steps": 0,
            "training_time": 0.0,
            "training_throughput": None,
        })

    nll = evaluate_nll(model, eval_loader, device, max_batches=2 if cfg.smoke else None)
    result.update(nll)
    result["relative_perplexity_change"] = relative_perplexity_change(
        result["perplexity"],
        baseline_perplexity,
    )
    result.update(downstream_accuracy(model, downstream_loader, device, max_batches=2 if cfg.smoke else None))

    generations = generation_outputs(
        model,
        tokenizer,
        cfg.generation_prompts,
        device,
        cfg.generation_max_new_tokens,
    )
    result["fixed_prompt_generations"] = generations
    result.update(repetition_diversity_stats(generations))
    result.update(latency_profile(
        model,
        tokenizer,
        cfg.generation_prompts[0],
        device,
        max_new_tokens=min(cfg.generation_max_new_tokens, 4 if cfg.smoke else cfg.generation_max_new_tokens),
    ))
    return result


def run_phase1(cfg):
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(cfg.model_name_or_path, revision=cfg.model_revision)
    datasets = make_lm_datasets(cfg, tokenizer)

    results = {
        "phase": "rq1_phase1_utility",
        "model_name_or_path": cfg.model_name_or_path,
        "model_revision": cfg.model_revision,
        "environment": environment_metadata(),
        "variants": [],
    }

    baseline_perplexity = None
    variants = list(cfg.variants)
    if Variant.ANN_ORIGINAL.value in variants:
        variants.remove(Variant.ANN_ORIGINAL.value)
        variants.insert(0, Variant.ANN_ORIGINAL.value)

    for variant in variants:
        variant_result = run_variant(cfg, variant, tokenizer, datasets, baseline_perplexity)
        if Variant(variant) == Variant.ANN_ORIGINAL:
            baseline_perplexity = variant_result["perplexity"]
            variant_result["relative_perplexity_change"] = 0.0
        results["variants"].append(variant_result)

    output_path = output_dir / "phase1_results.json"
    output_path.write_text(json.dumps(results, indent=2))
    return results
