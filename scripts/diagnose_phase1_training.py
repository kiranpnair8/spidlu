import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spidlu.config import load_config
from spidlu.data import load_tokenizer, make_dataloader, make_lm_datasets
from spidlu.layers import BlendedActivation
from spidlu.phase1 import build_variant_model
from spidlu.seed import set_seed
from spidlu.surgery import Variant, trainable_parameter_names
from spidlu.train import (
    clamp_trainable_blend_alphas,
    changed_trainable_parameters,
    gradient_norms,
    optimizer_parameter_group_summary,
    trainable_parameter_snapshot,
)


def activation_alpha_values(model):
    values = {}
    for name, module in model.named_modules():
        if isinstance(module, BlendedActivation):
            values[name] = {
                "raw": float(module.blend_alpha.detach().cpu().item()),
                "effective": module.alpha_value(),
                "requires_grad": bool(getattr(module.blend_alpha, "requires_grad", False)),
                "alpha_mode": module.alpha_mode,
                "alpha_max": module.alpha_max,
            }
    return values


def main():
    parser = argparse.ArgumentParser(description="Diagnose whether a Phase 1 variant has trainable parameters and gradients.")
    parser.add_argument("--config", default="configs/phase1_rq1_publication.yaml")
    parser.add_argument("--variant", default="spidlu", choices=[variant.value for variant in Variant])
    parser.add_argument("--seed", type=int, help="Override configured seed.")
    parser.add_argument("--output", help="Optional JSON output path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = load_tokenizer(cfg.model_name_or_path, revision=cfg.model_revision)
    datasets = make_lm_datasets(cfg, tokenizer)
    loader = make_dataloader(datasets[cfg.train_split], batch_size=cfg.batch_size, shuffle=True, seed=cfg.seed)
    batch = next(iter(loader))

    model, base_fingerprint, replacements = build_variant_model(cfg, args.variant, device)
    names = trainable_parameter_names(model)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = None
    if trainable_params:
        optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    before = trainable_parameter_snapshot(model)
    alpha_before = activation_alpha_values(model)

    input_ids = batch["input_ids"].to(device)
    labels = batch.get("labels", batch["input_ids"]).to(device)
    model.train()
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    output = model(input_ids=input_ids, labels=labels)
    loss = output.loss
    if optimizer is not None:
        loss.backward()
        norms = gradient_norms(model)
        optimizer.step()
        clamp_trainable_blend_alphas(model)
    else:
        norms = {}
    alpha_after = activation_alpha_values(model)
    changed = changed_trainable_parameters(model, before)

    payload = {
        "variant": args.variant,
        "seed": cfg.seed,
        "device": str(device),
        "base_weight_fingerprint": base_fingerprint,
        "replacement_records": [record.__dict__ for record in replacements],
        "trainable_parameter_names": names,
        "trainable_parameter_count": sum(param.numel() for param in trainable_params),
        "optimizer_parameter_groups": optimizer_parameter_group_summary(optimizer),
        "loss": float(loss.detach().cpu().item()),
        "gradient_norms": norms,
        "changed_trainable_parameters": changed,
        "any_trainable_parameter_changed": any(changed.values()) if changed else False,
        "alpha_before": alpha_before,
        "alpha_after": alpha_after,
    }

    text = json.dumps(payload, indent=2) + "\n"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
