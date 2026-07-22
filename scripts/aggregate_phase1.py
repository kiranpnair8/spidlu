import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


METRICS = [
    "perplexity",
    "token_weighted_nll",
    "relative_perplexity_change",
    "downstream_accuracy",
    "prefill_latency",
    "decode_latency_per_token",
    "prefill_throughput",
    "decode_throughput",
    "peak_cuda_allocated",
    "peak_cuda_reserved",
    "optimizer_steps",
    "processed_tokens",
    "training_time",
    "training_throughput",
    "parameters",
    "trainable_parameters",
]

RUN_COLUMNS = [
    "run_id",
    "timestamp",
    "mode",
    "run_dir",
    "model_name_or_path",
    "model_revision",
    "variant",
    "seed",
    *METRICS,
]

SUMMARY_COLUMNS = ["variant", "metric", "n", "mean", "std", "min", "max"]


def numeric_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def load_rows(input_root):
    rows = []
    for result_path in sorted(Path(input_root).rglob("phase1_results.json")):
        payload = json.loads(result_path.read_text())
        for variant in payload.get("variants", []):
            row = {
                "run_id": payload.get("run_id"),
                "timestamp": payload.get("timestamp"),
                "mode": payload.get("mode"),
                "run_dir": payload.get("run_dir", str(result_path.parent)),
                "model_name_or_path": payload.get("model_name_or_path"),
                "model_revision": payload.get("model_revision"),
                "variant": variant.get("variant"),
                "seed": variant.get("seed"),
            }
            for metric in METRICS:
                row[metric] = variant.get(metric)
            rows.append(row)
    return rows


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    variants = sorted({row["variant"] for row in rows if row.get("variant")})
    summary = []
    for variant in variants:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        for metric in METRICS:
            values = [numeric_or_none(row.get(metric)) for row in variant_rows]
            values = [value for value in values if value is not None]
            if not values:
                continue
            summary.append({
                "variant": variant,
                "metric": metric,
                "n": len(values),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
            })
    return summary


def plot_metric(summary_rows, metric, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rows = [row for row in summary_rows if row["metric"] == metric]
    if not rows:
        return None
    variants = [row["variant"] for row in rows]
    means = [float(row["mean"]) for row in rows]
    stds = [float(row["std"]) for row in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(variants)), 4))
    ax.bar(variants, means, yerr=stds, capsize=4)
    ax.set_ylabel(metric)
    ax.set_title(f"Phase 1 {metric} by variant")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    path = Path(output_dir) / f"{metric}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Aggregate Phase 1 RQ1 result directories.")
    parser.add_argument("--input-root", required=True, help="Root containing phase1_results.json files.")
    parser.add_argument("--output-dir", required=True, help="Directory for aggregate CSVs and plots.")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib plots.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rows = load_rows(args.input_root)
    if not rows:
        raise SystemExit(f"No phase1_results.json files found under {args.input_root}")
    summary_rows = summarize(rows)
    write_csv(output_dir / "phase1_runs.csv", rows, RUN_COLUMNS)
    write_csv(output_dir / "phase1_summary_mean_std.csv", summary_rows, SUMMARY_COLUMNS)
    metadata = {
        "input_root": str(args.input_root),
        "output_dir": str(output_dir),
        "num_variant_rows": len(rows),
        "variants": sorted({row["variant"] for row in rows if row.get("variant")}),
        "seeds": sorted({row["seed"] for row in rows if row.get("seed") is not None}),
    }
    (output_dir / "aggregation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.no_plots:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for metric in ("perplexity", "relative_perplexity_change", "downstream_accuracy", "decode_throughput"):
            plot_metric(summary_rows, metric, plot_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
