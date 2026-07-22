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

COMPARISON_COLUMNS = [
    "baseline_variant",
    "variant",
    "metric",
    "n",
    "mean_difference",
    "std_difference",
    "percent_change_mean",
    "t_statistic",
    "p_value",
    "p_value_method",
    "cohens_dz",
    "ci95_low",
    "ci95_high",
]

PUBLICATION_METRICS = [
    "perplexity",
    "relative_perplexity_change",
    "downstream_accuracy",
    "decode_latency_per_token",
    "decode_throughput",
    "peak_cuda_allocated",
    "peak_cuda_reserved",
    "processed_tokens",
    "optimizer_steps",
    "training_time",
]

BASELINE_VARIANTS = ("ann_original", "ann_compute_matched")

T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


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


def normal_two_sided_p_value(t_statistic):
    return math.erfc(abs(t_statistic) / math.sqrt(2.0))


def scipy_paired_p_value(baseline_values, variant_values):
    try:
        from scipy import stats
    except ImportError:
        return None
    result = stats.ttest_rel(variant_values, baseline_values, nan_policy="omit")
    if math.isnan(result.pvalue):
        return None
    return float(result.pvalue)


def paired_comparisons(rows, baselines=BASELINE_VARIANTS):
    comparisons = []
    variants = sorted({row["variant"] for row in rows if row.get("variant")})
    for baseline_variant in baselines:
        for variant in variants:
            if variant == baseline_variant:
                continue
            for metric in PUBLICATION_METRICS:
                baseline_by_seed = {}
                variant_by_seed = {}
                for row in rows:
                    seed = row.get("seed")
                    value = numeric_or_none(row.get(metric))
                    if seed is None or value is None:
                        continue
                    if row.get("variant") == baseline_variant:
                        baseline_by_seed[seed] = value
                    elif row.get("variant") == variant:
                        variant_by_seed[seed] = value
                paired_seeds = sorted(set(baseline_by_seed) & set(variant_by_seed))
                if not paired_seeds:
                    continue
                baseline_values = [baseline_by_seed[seed] for seed in paired_seeds]
                variant_values = [variant_by_seed[seed] for seed in paired_seeds]
                diffs = [variant_value - baseline_value for baseline_value, variant_value in zip(baseline_values, variant_values)]
                n = len(diffs)
                mean_diff = statistics.mean(diffs)
                std_diff = statistics.stdev(diffs) if n > 1 else 0.0
                baseline_mean = statistics.mean(baseline_values)
                percent_change = (mean_diff / baseline_mean * 100.0) if baseline_mean else None
                if n > 1 and std_diff > 0:
                    se = std_diff / math.sqrt(n)
                    t_statistic = mean_diff / se
                    p_value = scipy_paired_p_value(baseline_values, variant_values)
                    p_method = "paired_t_test_scipy" if p_value is not None else "paired_t_test_normal_approx"
                    if p_value is None:
                        p_value = normal_two_sided_p_value(t_statistic)
                    cohens_dz = mean_diff / std_diff
                    critical = T_CRITICAL_975.get(n - 1, 1.96)
                    ci95_low = mean_diff - critical * se
                    ci95_high = mean_diff + critical * se
                else:
                    t_statistic = None
                    p_value = None
                    p_method = "insufficient_paired_samples"
                    cohens_dz = None
                    ci95_low = None
                    ci95_high = None
                comparisons.append({
                    "baseline_variant": baseline_variant,
                    "variant": variant,
                    "metric": metric,
                    "n": n,
                    "mean_difference": mean_diff,
                    "std_difference": std_diff,
                    "percent_change_mean": percent_change,
                    "t_statistic": t_statistic,
                    "p_value": p_value,
                    "p_value_method": p_method,
                    "cohens_dz": cohens_dz,
                    "ci95_low": ci95_low,
                    "ci95_high": ci95_high,
                })
    return comparisons


def publication_table(summary_rows, comparison_rows, baseline="ann_original"):
    summary_by_key = {(row["variant"], row["metric"]): row for row in summary_rows}
    comparison_by_key = {
        (row["variant"], row["metric"]): row
        for row in comparison_rows
        if row["baseline_variant"] == baseline
    }
    variants = sorted({row["variant"] for row in summary_rows if row.get("variant")})
    lines = [
        "| Variant | Metric | Mean | Std | N | Delta vs ann_original | p-value |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in variants:
        for metric in PUBLICATION_METRICS:
            row = summary_by_key.get((variant, metric))
            if row is None:
                continue
            comp = comparison_by_key.get((variant, metric), {})
            lines.append(
                "| {variant} | {metric} | {mean:.6g} | {std:.6g} | {n} | {delta} | {p_value} |".format(
                    variant=variant,
                    metric=metric,
                    mean=float(row["mean"]),
                    std=float(row["std"]),
                    n=row["n"],
                    delta="" if variant == baseline else format_optional(comp.get("mean_difference")),
                    p_value="" if variant == baseline else format_optional(comp.get("p_value")),
                )
            )
    return "\n".join(lines) + "\n"


def format_optional(value):
    value = numeric_or_none(value)
    if value is None:
        return ""
    return f"{value:.6g}"


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


def plot_variant_metric_runs(rows, metric, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    variants = sorted({row["variant"] for row in rows if row.get("variant") and numeric_or_none(row.get(metric)) is not None})
    if not variants:
        return None
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(variants)), 4))
    for index, variant in enumerate(variants):
        values = [numeric_or_none(row.get(metric)) for row in rows if row.get("variant") == variant]
        values = [value for value in values if value is not None]
        ax.scatter([index] * len(values), values, alpha=0.8)
    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=25, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Phase 1 per-seed {metric}")
    fig.tight_layout()
    path = Path(output_dir) / f"{metric}_per_seed.png"
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
    comparison_rows = paired_comparisons(rows)
    write_csv(output_dir / "phase1_runs.csv", rows, RUN_COLUMNS)
    write_csv(output_dir / "phase1_summary_mean_std.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(output_dir / "phase1_paired_significance.csv", comparison_rows, COMPARISON_COLUMNS)
    (output_dir / "phase1_publication_table.md").write_text(publication_table(summary_rows, comparison_rows))
    metadata = {
        "input_root": str(args.input_root),
        "output_dir": str(output_dir),
        "num_variant_rows": len(rows),
        "variants": sorted({row["variant"] for row in rows if row.get("variant")}),
        "seeds": sorted({row["seed"] for row in rows if row.get("seed") is not None}),
        "paired_comparisons": len(comparison_rows),
        "statistical_test": "paired t-test by seed when scipy is available; otherwise normal approximation to paired t-statistic",
    }
    (output_dir / "aggregation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not args.no_plots:
        plot_dir = output_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for metric in ("perplexity", "relative_perplexity_change", "downstream_accuracy", "decode_throughput"):
            plot_metric(summary_rows, metric, plot_dir)
            plot_variant_metric_runs(rows, metric, plot_dir)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
