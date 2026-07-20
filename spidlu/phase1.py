"""End-to-end Phase 1 RQ1 runner."""

import csv
import hashlib
import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from spidlu.data import load_tokenizer, make_dataloader, make_lm_datasets
from spidlu.env import require_huggingface_runtime
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

SUMMARY_COLUMNS = [
    "variant",
    "seed",
    "run_id",
    "timestamp",
    "mode",
    "checkpoint_path",
    "perplexity",
    "downstream_accuracy",
    "prefill_latency",
    "decode_latency_per_token",
    "prefill_throughput",
    "decode_throughput",
    "training_throughput",
    "peak_cuda_allocated",
    "peak_cuda_reserved",
    "optimizer_steps",
    "processed_tokens",
    "training_time",
    "run_dir",
]


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path, payload):
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def atomic_write_yaml(path, payload):
    atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False))


def build_run_context(cfg, run_id=None, overwrite=False):
    timestamp = utc_timestamp()
    resolved_run_id = run_id or f"{timestamp}_{uuid.uuid4().hex[:8]}"
    mode = "smoke" if cfg.smoke else "full"
    variant_slug = "-".join(cfg.variants) if len(cfg.variants) <= 2 else f"multi{len(cfg.variants)}"
    run_name = (
        f"{variant_slug}_seed{cfg.seed}_{resolved_run_id}"
        if run_id
        else f"{variant_slug}_seed{cfg.seed}_{timestamp}_{resolved_run_id}"
    )
    output_root = Path(cfg.output_dir)
    run_dir = output_root / mode / run_name
    if run_dir.exists() and not overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir}. Use --overwrite to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": resolved_run_id,
        "timestamp": timestamp,
        "mode": mode,
        "run_dir": run_dir,
        "summary_path": output_root / f"phase1_{mode}_summary.csv",
    }


def append_summary_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    rows.append({column: row.get(column) for column in SUMMARY_COLUMNS})
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def summary_row(result, run_context):
    return {
        "variant": result.get("variant"),
        "seed": result.get("seed"),
        "run_id": run_context["run_id"],
        "timestamp": run_context["timestamp"],
        "mode": run_context["mode"],
        "checkpoint_path": result.get("checkpoint_path"),
        "perplexity": result.get("perplexity"),
        "downstream_accuracy": result.get("downstream_accuracy"),
        "prefill_latency": result.get("prefill_latency"),
        "decode_latency_per_token": result.get("decode_latency_per_token"),
        "prefill_throughput": result.get("prefill_throughput"),
        "decode_throughput": result.get("decode_throughput"),
        "training_throughput": result.get("training_throughput"),
        "peak_cuda_allocated": result.get("peak_cuda_allocated"),
        "peak_cuda_reserved": result.get("peak_cuda_reserved"),
        "optimizer_steps": result.get("optimizer_steps"),
        "processed_tokens": result.get("processed_tokens"),
        "training_time": result.get("training_time"),
        "run_dir": str(run_context["run_dir"]),
    }


def load_causal_lm(cfg):
    require_huggingface_runtime()
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


def run_variant(cfg, variant, tokenizer, datasets, baseline_perplexity=None, checkpoint_dir=None):
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
        "seed": cfg.seed,
        "base_weight_fingerprint": base_fingerprint,
        "replacement_records": [record.__dict__ for record in replacements],
        **count_parameters(model),
    }

    if variant in TRAINED_VARIANTS:
        result.update(train_variant(model, train_loader, cfg, device, checkpoint_dir=checkpoint_dir))
    else:
        result.update({
            "processed_tokens": 0,
            "optimizer_steps": 0,
            "training_time": 0.0,
            "training_throughput": None,
            "checkpoint_path": None,
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


def run_phase1(cfg, run_id=None, overwrite=False):
    set_seed(cfg.seed)
    run_context = build_run_context(cfg, run_id=run_id, overwrite=overwrite)
    run_dir = run_context["run_dir"]
    tokenizer = load_tokenizer(cfg.model_name_or_path, revision=cfg.model_revision)
    datasets = make_lm_datasets(cfg, tokenizer)

    results = {
        "phase": "rq1_phase1_utility",
        "run_id": run_context["run_id"],
        "timestamp": run_context["timestamp"],
        "mode": run_context["mode"],
        "run_dir": str(run_dir),
        "model_name_or_path": cfg.model_name_or_path,
        "model_revision": cfg.model_revision,
        "environment": environment_metadata(),
        "variants": [],
    }

    metadata = {
        "run_id": run_context["run_id"],
        "timestamp": run_context["timestamp"],
        "mode": run_context["mode"],
        "run_dir": str(run_dir),
        "summary_path": str(run_context["summary_path"]),
        "model_name_or_path": cfg.model_name_or_path,
        "model_revision": cfg.model_revision,
        "environment": results["environment"],
    }
    atomic_write_yaml(run_dir / "resolved_config.yaml", asdict(cfg))
    atomic_write_json(run_dir / "metadata.json", metadata)

    baseline_perplexity = None
    summary_rows = []
    variants = list(cfg.variants)
    if Variant.ANN_ORIGINAL.value in variants:
        variants.remove(Variant.ANN_ORIGINAL.value)
        variants.insert(0, Variant.ANN_ORIGINAL.value)

    for variant in variants:
        checkpoint_dir = run_dir / "checkpoints" / variant if Variant(variant) in TRAINED_VARIANTS else None
        variant_result = run_variant(
            cfg,
            variant,
            tokenizer,
            datasets,
            baseline_perplexity,
            checkpoint_dir=checkpoint_dir,
        )
        if Variant(variant) == Variant.ANN_ORIGINAL:
            baseline_perplexity = variant_result["perplexity"]
            variant_result["relative_perplexity_change"] = 0.0
        results["variants"].append(variant_result)
        summary_rows.append(summary_row(variant_result, run_context))

    generations = [
        {"variant": item["variant"], **generation}
        for item in results["variants"]
        for generation in item.get("fixed_prompt_generations", [])
    ]
    if generations:
        atomic_write_text(
            run_dir / "generations.jsonl",
            "".join(json.dumps(item) + "\n" for item in generations),
        )

    atomic_write_json(run_dir / "phase1_results.json", results)
    for row in summary_rows:
        append_summary_row(run_context["summary_path"], row)
    return results
