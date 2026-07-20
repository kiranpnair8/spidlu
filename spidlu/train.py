"""Training loop for Phase 1 trained variants."""

import time
from pathlib import Path

import torch

try:
    from spikingjelly.activation_based import functional
except ImportError:  # pragma: no cover - only for lightweight smoke imports.
    class _FunctionalFallback:
        @staticmethod
        def reset_net(model):
            return None

    functional = _FunctionalFallback()


def train_variant(model, dataloader, cfg, device, checkpoint_dir=None):
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
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
            optimizer.zero_grad(set_to_none=True)
            functional.reset_net(model)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            processed_tokens += labels.numel()
            optimizer_steps += 1
            if checkpoint_dir is not None and save_every_steps and optimizer_steps % save_every_steps == 0:
                checkpoint_path = checkpoint_dir / f"step_{optimizer_steps:06d}.pt"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "optimizer_steps": optimizer_steps,
                        "processed_tokens": processed_tokens,
                    },
                    checkpoint_path,
                )
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
