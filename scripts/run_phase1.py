import argparse

from spidlu.config import load_config
from spidlu.phase1 import run_phase1


def main():
    parser = argparse.ArgumentParser(description="Run RQ1 Phase 1 utility experiments.")
    parser.add_argument("--config", default="configs/phase1_rq1.yaml")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny end-to-end smoke pass.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg.smoke = True
        cfg.max_train_steps = min(cfg.max_train_steps, 1)
        cfg.generation_max_new_tokens = min(cfg.generation_max_new_tokens, 4)
    run_phase1(cfg)


if __name__ == "__main__":
    main()
