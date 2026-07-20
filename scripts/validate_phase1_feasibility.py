import argparse
import sys
from tempfile import TemporaryDirectory
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


VARIANTS = (
    "ann_original",
    "spidlu",
    "ann_compute_matched",
    "quantized_activation",
)
TRAINED = {
    "spidlu",
    "ann_compute_matched",
    "quantized_activation",
}


def main():
    parser = argparse.ArgumentParser(description="Validate Phase 1 feasibility setup.")
    parser.add_argument("--config", default="configs/phase1_rq1_feasibility.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="models/phase1_rq1_feasibility")
    parser.add_argument("--run-prefix", default="phase1_feasibility_seed42")
    args = parser.parse_args()

    from spidlu.config import load_config
    from spidlu.phase1 import TRAINED_VARIANTS, build_run_context

    cfg = load_config(args.config)
    assert cfg.smoke is False, "Feasibility config must not use smoke mode."
    assert cfg.seed == args.seed, f"Expected seed {args.seed}, found {cfg.seed}."
    assert cfg.variants == list(VARIANTS), f"Unexpected variants: {cfg.variants}"
    assert cfg.max_train_steps == 8, f"Expected 8 feasibility steps, found {cfg.max_train_steps}."
    assert cfg.max_train_tokens == 4096, f"Expected 4096 feasibility tokens, found {cfg.max_train_tokens}."

    trained_values = {variant.value for variant in TRAINED_VARIANTS}
    assert "ann_original" not in trained_values, "ann_original must remain evaluation-only."
    assert trained_values == TRAINED, f"Unexpected trained variants: {trained_values}"

    shared = {
        "seed": cfg.seed,
        "dataset_name": cfg.dataset_name,
        "dataset_config": cfg.dataset_config,
        "train_split": cfg.train_split,
        "eval_split": cfg.eval_split,
        "downstream_split": cfg.downstream_split,
        "batch_size": cfg.batch_size,
        "eval_batch_size": cfg.eval_batch_size,
        "max_seq_len": cfg.max_seq_len,
        "max_train_steps": cfg.max_train_steps,
        "max_train_tokens": cfg.max_train_tokens,
    }

    planned_dirs = []
    with TemporaryDirectory() as tmpdir:
        run_dirs = []
        for variant in VARIANTS:
            variant_cfg = load_config(args.config)
            variant_cfg.variants = [variant]
            variant_cfg.seed = args.seed
            variant_cfg.output_dir = str(Path(tmpdir) / variant)
            context = build_run_context(
                variant_cfg,
                run_id=f"{args.run_prefix}_{variant}",
                overwrite=True,
            )
            run_dirs.append(context["run_dir"])
            planned_dirs.append(
                Path(args.output_root)
                / variant
                / "full"
                / f"{variant}_seed{args.seed}_{args.run_prefix}_{variant}"
            )
            for key, value in shared.items():
                assert getattr(variant_cfg, key) == value, f"{variant} changed shared setting {key}."

    assert len(set(run_dirs)) == len(VARIANTS), f"Output paths collide: {run_dirs}"
    assert len(set(planned_dirs)) == len(VARIANTS), f"Planned output paths collide: {planned_dirs}"
    print("Phase 1 feasibility validation passed.")
    print(f"Shared seed: {args.seed}")
    print(f"Training budget for trained variants: {cfg.max_train_steps} steps / {cfg.max_train_tokens} tokens")
    for variant, run_dir in zip(VARIANTS, planned_dirs):
        print(f"{variant}: {run_dir}")


if __name__ == "__main__":
    main()
