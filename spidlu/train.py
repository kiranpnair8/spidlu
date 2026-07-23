"""Training loop for Phase 1 trained variants."""

import time
from pathlib import Path

import torch

from spidlu.layers import BlendedActivation

try:
    from spikingjelly.activation_based import functional
except ImportError:  # pragma: no cover - only for lightweight smoke imports.
    class _FunctionalFallback:
        @staticmethod
        def reset_net(model):
            return None

    functional = _FunctionalFallback()


def blended_activation_modules(model):
    for module in model.modules():
        if isinstance(module, BlendedActivation):
            yield module


def clamp_trainable_blend_alphas(model):
    for module in blended_activation_modules(model):
        if isinstance(module.blend_alpha, torch.nn.Parameter):
            module.set_alpha(module.blend_alpha.detach().item())


def trainable_parameter_snapshot(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def gradient_norms(model):
    norms = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            norms[name] = None if param.grad is None else float(param.grad.detach().norm().cpu().item())
    return norms


def changed_trainable_parameters(model, before):
    changed = {}
    for name, param in model.named_parameters():
        if name in before:
            changed[name] = not torch.equal(before[name], param.detach())
    return changed


def optimizer_parameter_group_summary(optimizer):
    if optimizer is None:
        return []
    return [
        {
            "group_index": idx,
            "parameter_count": len(group["params"]),
            "element_count": sum(param.numel() for param in group["params"]),
            "lr": group.get("lr"),
            "weight_decay": group.get("weight_decay"),
        }
        for idx, group in enumerate(optimizer.param_groups)
    ]


def set_linear_warmup_alpha(model, cfg, optimizer_steps):
    alpha_mode = getattr(cfg, "spidlu_alpha_mode", "trainable")
    if alpha_mode != "linear_warmup":
        return
    alpha_max = getattr(cfg, "spidlu_alpha_max", 0.1)
    warmup_steps = getattr(cfg, "spidlu_warmup_steps", None) or max(1, cfg.max_train_steps)
    alpha = alpha_max * min(1.0, optimizer_steps / max(1, warmup_steps))
    for module in blended_activation_modules(model):
        module.set_alpha(alpha)


def train_variant(model, dataloader, cfg, device, checkpoint_dir=None):
    model.train()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = None
    scheduler = None
    if trainable_params:
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    set_linear_warmup_alpha(model, cfg, 0)
    processed_tokens = 0
    optimizer_steps = 0
    checkpoint_path = None
    save_every_steps = getattr(cfg, "save_every_steps", None)
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    while optimizer_steps < cfg.max_train_steps:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch.get("labels", batch["input_ids"]).to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            functional.reset_net(model)
            if optimizer is not None:
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                clamp_trainable_blend_alphas(model)
                scheduler.step()
            else:
                with torch.inference_mode():
                    model(input_ids=input_ids, labels=labels)
            processed_tokens += labels.numel()
            optimizer_steps += 1
            set_linear_warmup_alpha(model, cfg, optimizer_steps)
            if checkpoint_dir is not None and save_every_steps and optimizer_steps % save_every_steps == 0:
                checkpoint_path = checkpoint_dir / f"step_{optimizer_steps:06d}.pt"
                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer_steps": optimizer_steps,
                    "processed_tokens": processed_tokens,
                }
                if optimizer is not None:
                    checkpoint["optimizer"] = optimizer.state_dict()
                    checkpoint["scheduler"] = scheduler.state_dict()
                torch.save(checkpoint, checkpoint_path)
            if cfg.max_train_tokens is not None and processed_tokens >= cfg.max_train_tokens:
                break
            if optimizer_steps >= cfg.max_train_steps:
                break
        if cfg.max_train_tokens is not None and processed_tokens >= cfg.max_train_tokens:
            break

    elapsed = time.perf_counter() - start
    return {
        "processed_tokens": processed_tokens,
        "optimizer_steps": optimizer_steps,
        "training_time": elapsed,
        "training_throughput": processed_tokens / max(elapsed, 1e-9),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
    }
