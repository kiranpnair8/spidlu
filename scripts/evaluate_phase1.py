import argparse

from spidlu.config import load_config
from spidlu.phase1 import run_phase1


def main():
    parser = argparse.ArgumentParser(description="Evaluate Phase 1 variants.")
    parser.add_argument("--config", default="configs/phase1_rq1.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.smoke:
        cfg.smoke = True
        cfg.max_train_steps = 0
        cfg.generation_max_new_tokens = min(cfg.generation_max_new_tokens, 4)
    run_phase1(cfg)


if __name__ == "__main__":
    main()
