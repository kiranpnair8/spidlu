import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 variants.")
    parser.add_argument("--config", default="configs/phase1_rq1.yaml")
    parser.add_argument(
        "--variant",
        choices=("ann_original", "spidlu", "ann_compute_matched", "quantized_activation"),
        help="Evaluate a single Phase 1 variant instead of the full configured set.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    from spidlu.config import load_config
    from spidlu.phase1 import run_phase1

    cfg = load_config(args.config)
    if args.variant:
        cfg.variants = [args.variant]
    if args.smoke:
        cfg.smoke = True
        cfg.max_train_steps = 0
        cfg.generation_max_new_tokens = min(cfg.generation_max_new_tokens, 4)
    run_phase1(cfg)


if __name__ == "__main__":
    main()
