import argparse
import copy
import gc
import json
import math
import sys
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def tensor_stats(x):
    flat = x.detach().float().reshape(-1)
    return {
        "mean": flat.mean().item(),
        "std": flat.std(unbiased=False).item(),
        "min": flat.min().item(),
        "max": flat.max().item(),
        "zero_fraction": flat.eq(0).float().mean().item(),
        "positive_fraction": flat.gt(0).float().mean().item(),
        "l1_mean_abs": flat.abs().mean().item(),
    }


def compare_to_reference(output, reference):
    import torch
    import torch.nn.functional as F

    out_flat = output.detach().float().reshape(1, -1)
    ref_flat = reference.detach().float().reshape(1, -1)
    return {
        "cosine_similarity_to_silu": F.cosine_similarity(out_flat, ref_flat).item(),
        "mean_absolute_error_from_silu": torch.mean(torch.abs(output.detach().float() - reference.detach().float())).item(),
    }


def cleanup_model(model):
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clone_cfg(cfg, **updates):
    return replace(copy.deepcopy(cfg), **updates)


def capture_activation_input(model, batch, device, layer_index):
    import torch

    from spidlu.surgery import _activation_attr, iter_transformer_mlps

    mlps = list(iter_transformer_mlps(model))
    if layer_index < 0 or layer_index >= len(mlps):
        raise ValueError(f"layer_index {layer_index} is outside [0, {len(mlps) - 1}].")
    _, module_path, mlp = mlps[layer_index]
    activation = getattr(mlp, _activation_attr(mlp))
    captured = {}

    def hook(_module, inputs):
        captured["x"] = inputs[0].detach().cpu()

    handle = activation.register_forward_pre_hook(hook)
    try:
        model.eval()
        with torch.inference_mode():
            kwargs = {"input_ids": batch["input_ids"].to(device)}
            if "attention_mask" in batch:
                kwargs["attention_mask"] = batch["attention_mask"].to(device)
            model(**kwargs)
    finally:
        handle.remove()
    if "x" not in captured:
        raise RuntimeError(f"Did not capture activation input for {module_path}.")
    return module_path, captured["x"]


def activation_comparison(cfg, batch, device, representative_layer):
    import torch
    import torch.nn as nn

    from spidlu.layers import BlendedActivation, QuantizedActivationSTE, SpiDLU
    from spidlu.phase1 import build_variant_model
    from spidlu.surgery import Variant

    model, fingerprint, replacements = build_variant_model(
        clone_cfg(cfg, surgery_scope="all"),
        Variant.ANN_ORIGINAL,
        device,
    )
    module_path, activation_input = capture_activation_input(model, batch, device, representative_layer)
    cleanup_model(model)

    x = activation_input.to(device)
    silu = nn.SiLU().to(device)
    spidlu = SpiDLU(alpha=cfg.spidlu_alpha, threshold=cfg.spidlu_threshold, T=cfg.spidlu_T).to(device)
    quantized = QuantizedActivationSTE(base_activation=nn.SiLU(), levels=cfg.quantized_levels or (cfg.spidlu_T + 1)).to(device)
    blended_zero = BlendedActivation(
        nn.SiLU(),
        SpiDLU(alpha=cfg.spidlu_alpha, threshold=cfg.spidlu_threshold, T=cfg.spidlu_T),
        blend_alpha=0.0,
        trainable=False,
    ).to(device)

    with torch.inference_mode():
        outputs = {
            "silu": silu(x),
            "spidlu": spidlu(x),
            "quantized_activation": quantized(x),
            "spidlu_blend_alpha0": blended_zero(x),
        }
    reference = outputs["silu"]
    comparisons = {}
    for name, output in outputs.items():
        stats = tensor_stats(output)
        if name == "silu":
            stats.update({
                "cosine_similarity_to_silu": 1.0,
                "mean_absolute_error_from_silu": 0.0,
            })
        else:
            stats.update(compare_to_reference(output, reference))
        stats["gate_or_spike_activation_rate"] = stats["positive_fraction"]
        comparisons[name] = stats

    return {
        "base_weight_fingerprint": fingerprint,
        "replacement_records": [record.__dict__ for record in replacements],
        "representative_layer_index": representative_layer,
        "representative_module_path": module_path,
        "input_stats": tensor_stats(x),
        "outputs": comparisons,
    }


def evaluate_zero_step(cfg, variant_name, surgery_scope, batch, device, layer_index=None, first_n=None):
    import torch

    from spidlu.eval import compare_hf_loss
    from spidlu.metrics import count_parameters
    from spidlu.phase1 import build_variant_model, state_fingerprint
    from spidlu.surgery import Variant

    scoped_cfg = clone_cfg(
        cfg,
        surgery_scope=surgery_scope,
        surgery_layer_index=layer_index,
        surgery_first_n=first_n,
    )
    model, fingerprint, replacements = build_variant_model(scoped_cfg, Variant(variant_name), device)
    before = state_fingerprint(model)
    model.eval()
    with torch.inference_mode():
        input_ids = batch["input_ids"].to(device)
        labels = batch.get("labels", batch["input_ids"]).to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        comparison = compare_hf_loss(
            model,
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
    after = state_fingerprint(model)
    result = {
        "variant": variant_name,
        "surgery_scope": surgery_scope,
        "surgery_layer_index": layer_index,
        "surgery_first_n": first_n,
        "optimizer_steps": 0,
        "causal_lm_loss": comparison["custom_loss"].item(),
        "hf_loss": comparison["hf_loss"].item(),
        "perplexity": math.exp(comparison["custom_loss"].item()),
        "valid_token_count": comparison["valid_tokens"].item(),
        "base_weight_fingerprint": fingerprint,
        "post_construction_fingerprint": before,
        "post_eval_fingerprint": after,
        "weights_changed_during_diagnostic": before != after,
        "replacement_count": len(replacements),
        "replaced_modules": [f"{record.module_path}.{record.activation_attr}" for record in replacements],
        "replacement_records": [record.__dict__ for record in replacements],
        **count_parameters(model),
    }
    cleanup_model(model)
    return result


def main():
    parser = argparse.ArgumentParser(description="Diagnose zero-step Phase 1 activation surgery effects.")
    parser.add_argument("--config", default="configs/phase1_rq1_feasibility.yaml")
    parser.add_argument("--output-dir", default="models/phase1_surgery_diagnostics")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch

    from spidlu.config import load_config
    from spidlu.data import load_tokenizer, make_dataloader, make_lm_datasets
    from spidlu.phase1 import atomic_write_yaml, load_causal_lm
    from spidlu.surgery import iter_transformer_mlps

    cfg = load_config(args.config)
    timestamp = utc_timestamp()
    run_id = args.run_id or f"{timestamp}_{uuid.uuid4().hex[:8]}"
    run_dir = Path(args.output_dir) / run_id
    if run_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Diagnostic directory already exists: {run_dir}. Use --overwrite to replace it.")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg.model_name_or_path, revision=cfg.model_revision)
    datasets = make_lm_datasets(cfg, tokenizer)
    eval_loader = make_dataloader(
        datasets[cfg.eval_split],
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        seed=cfg.seed,
    )
    fixed_batch = next(iter(eval_loader))

    probe_model = load_causal_lm(cfg)
    layer_count = len(list(iter_transformer_mlps(probe_model)))
    cleanup_model(probe_model)
    middle_layer = layer_count // 2

    diagnostic_specs = [
        ("ann_original", "all", None, None),
        ("ann_compute_matched", "all", None, None),
        ("spidlu", "all", None, None),
        ("quantized_activation", "all", None, None),
        ("spidlu", "one", middle_layer, None),
    ]
    evaluations = [
        evaluate_zero_step(cfg, variant, scope, fixed_batch, device, layer_index=layer, first_n=first_n)
        for variant, scope, layer, first_n in diagnostic_specs
    ]
    activation = activation_comparison(cfg, fixed_batch, device, middle_layer)

    result = {
        "phase": "rq1_phase1_model_surgery_diagnostic",
        "timestamp": timestamp,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "model_name_or_path": cfg.model_name_or_path,
        "model_revision": cfg.model_revision,
        "seed": cfg.seed,
        "device": str(device),
        "eval_split": cfg.eval_split,
        "eval_batch_size": cfg.eval_batch_size,
        "max_seq_len": cfg.max_seq_len,
        "fixed_batch": {
            "input_shape": list(fixed_batch["input_ids"].shape),
            "input_id_min": int(fixed_batch["input_ids"].min().item()),
            "input_id_max": int(fixed_batch["input_ids"].max().item()),
            "decoded_prefix": tokenizer.decode(fixed_batch["input_ids"][0, :128].tolist(), skip_special_tokens=False),
        },
        "layer_count": layer_count,
        "representative_layer_index": middle_layer,
        "zero_step_evaluations": evaluations,
        "activation_comparison": activation,
    }

    atomic_write_yaml(run_dir / "resolved_config.yaml", asdict(cfg))
    atomic_write_json(run_dir / "surgery_diagnostic.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
