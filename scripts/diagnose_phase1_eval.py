import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def decode_sequences(tokenizer, input_ids, max_sequences=3):
    decoded = []
    for row in input_ids[:max_sequences]:
        decoded.append(tokenizer.decode(row.detach().cpu().tolist(), skip_special_tokens=False))
    return decoded


def main():
    parser = argparse.ArgumentParser(description="Diagnose Phase 1 causal-LM evaluation batches.")
    parser.add_argument("--config", default="configs/phase1_rq1_feasibility.yaml")
    parser.add_argument(
        "--variant",
        choices=("ann_original", "spidlu", "ann_compute_matched", "quantized_activation"),
        default="ann_original",
    )
    parser.add_argument("--batches", type=int, default=2)
    args = parser.parse_args()

    from spidlu.config import load_config
    from spidlu.data import load_tokenizer, make_dataloader, make_lm_datasets
    from spidlu.eval import compare_hf_loss, downstream_accuracy
    from spidlu.phase1 import build_variant_model
    from spidlu.surgery import Variant

    cfg = load_config(args.config)
    cfg.variants = [args.variant]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer(cfg.model_name_or_path, revision=cfg.model_revision)
    datasets = make_lm_datasets(cfg, tokenizer)
    eval_loader = make_dataloader(
        datasets[cfg.eval_split],
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        seed=cfg.seed,
    )
    model, fingerprint, replacements = build_variant_model(cfg, Variant(args.variant), device)
    model.eval()

    print(f"variant={args.variant}")
    print(f"base_fingerprint={fingerprint}")
    print(f"replacement_records={[record.__dict__ for record in replacements]}")
    print(f"tokenizer={type(tokenizer).__name__}")
    print(f"vocab_size={len(tokenizer)} model_vocab={getattr(model.config, 'vocab_size', None)}")
    print(f"padding_side={tokenizer.padding_side} eos_id={tokenizer.eos_token_id} pad_id={tokenizer.pad_token_id}")

    with torch.inference_mode():
        for batch_idx, batch in enumerate(eval_loader):
            if batch_idx >= args.batches:
                break
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
            logits = comparison["logits"]
            print(f"\nBATCH {batch_idx}")
            print(f"custom_loss={comparison['custom_loss'].item():.6f}")
            print(f"hf_loss={comparison['hf_loss'].item():.6f}")
            print(f"valid_token_count={comparison['valid_tokens'].item()}")
            print(f"input_shape={tuple(input_ids.shape)} logits_shape={tuple(logits.shape)} labels_shape={tuple(labels.shape)}")
            print(f"input_id_range=({input_ids.min().item()}, {input_ids.max().item()})")
            if attention_mask is not None:
                print(f"attention_mask_shape={tuple(attention_mask.shape)} active_tokens={attention_mask.sum().item()}")
            print(f"first_input_ids={input_ids[0, :32].detach().cpu().tolist()}")
            print(f"first_labels={labels[0, :32].detach().cpu().tolist()}")
            for seq_idx, text in enumerate(decode_sequences(tokenizer, input_ids)):
                print(f"decoded[{seq_idx}]={text[:500]!r}")

    downstream = downstream_accuracy(model, eval_loader, device, max_batches=args.batches)
    print("\nDOWNSTREAM DIAGNOSTIC")
    for key, value in downstream.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
