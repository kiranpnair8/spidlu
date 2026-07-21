import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(description="Run RQ1 Phase 1 utility experiments.")
    parser.add_argument("--config", default="configs/phase1_rq1.yaml")
    parser.add_argument(
        "--variant",
        choices=("ann_original", "spidlu", "ann_compute_matched", "quantized_activation"),
        help="Run a single Phase 1 variant instead of the full configured set.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end smoke pass.")
    parser.add_argument("--run-id", help="Optional stable run id to include in output paths.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into an existing run directory.")
    parser.add_argument("--seed", type=int, help="Override the configured random seed.")
    parser.add_argument("--output-dir", help="Override the configured output directory.")
    parser.add_argument("--max-train-steps", type=int, help="Override max_train_steps.")
    parser.add_argument("--max-train-tokens", type=int, help="Override max_train_tokens.")
    parser.add_argument("--surgery-scope", choices=("all", "one", "first_n"), help="Override activation surgery scope.")
    parser.add_argument("--surgery-layer-index", type=int, help="Layer index for --surgery-scope one.")
    parser.add_argument("--surgery-first-n", type=int, help="Layer count for --surgery-scope first_n.")
    parser.add_argument(
        "--spidlu-alpha-mode",
        choices=("trainable", "fixed", "linear_warmup"),
        help="Override Spi-DLU blend alpha mode.",
    )
    parser.add_argument("--spidlu-alpha-max", type=float, help="Override Spi-DLU blend alpha max.")
    parser.add_argument("--spidlu-fixed-alpha", type=float, help="Override fixed Spi-DLU blend alpha.")
    parser.add_argument("--spidlu-warmup-steps", type=int, help="Override linear warmup steps.")
    args = parser.parse_args()

    from spidlu.config import load_config
    from spidlu.phase1 import run_phase1

    cfg = load_config(args.config)
    if args.variant:
        cfg.variants = [args.variant]
    if args.seed is not None:
        cfg.seed = args.seed
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.max_train_steps is not None:
        cfg.max_train_steps = args.max_train_steps
    if args.max_train_tokens is not None:
        cfg.max_train_tokens = args.max_train_tokens
    if args.surgery_scope:
        cfg.surgery_scope = args.surgery_scope
    if args.surgery_layer_index is not None:
        cfg.surgery_layer_index = args.surgery_layer_index
    if args.surgery_first_n is not None:
        cfg.surgery_first_n = args.surgery_first_n
    if args.spidlu_alpha_mode:
        cfg.spidlu_alpha_mode = args.spidlu_alpha_mode
    if args.spidlu_alpha_max is not None:
        cfg.spidlu_alpha_max = args.spidlu_alpha_max
    if args.spidlu_fixed_alpha is not None:
        cfg.spidlu_fixed_alpha = args.spidlu_fixed_alpha
    if args.spidlu_warmup_steps is not None:
        cfg.spidlu_warmup_steps = args.spidlu_warmup_steps
    if args.smoke:
        cfg.smoke = True
        cfg.max_train_steps = min(cfg.max_train_steps, 1)
        cfg.generation_max_new_tokens = min(cfg.generation_max_new_tokens, 4)
    run_phase1(cfg, run_id=args.run_id, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
