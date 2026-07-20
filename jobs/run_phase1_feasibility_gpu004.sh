#!/bin/bash
# Feasibility run for RQ1 Phase 1 on gpu004.

#SBATCH --job-name=SpiDLU_RQ1_Feas
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu004
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/phase1_stage2/%x_%j.out
#SBATCH --error=logs/phase1_stage2/%x_%j.err

set -euo pipefail

resolve_project_root() {
    local expected="configs/phase1_rq1_feasibility.yaml"
    local candidates=(
        "${PROJECT_ROOT:-}"
        "${SLURM_SUBMIT_DIR:-}"
        "$PWD"
        "$(dirname "${BASH_SOURCE[0]}")/.."
    )
    local candidate
    for candidate in "${candidates[@]}"; do
        [[ -n "$candidate" ]] || continue
        if [[ -f "$candidate/$expected" ]]; then
            cd "$candidate" && pwd
            return 0
        fi
        if [[ -f "$candidate/../$expected" ]]; then
            cd "$candidate/.." && pwd
            return 0
        fi
    done
    echo "Could not locate repository root containing $expected." >&2
    echo "Submit from the repo root or run: PROJECT_ROOT=/path/to/spidlu sbatch jobs/run_phase1_feasibility_gpu004.sh" >&2
    return 1
}

PROJECT_ROOT="$(resolve_project_root)"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/phase1_rq1_feasibility.yaml}"
CONDA_ENV="${CONDA_ENV:-spidlu}"
CONDA_SH="${CONDA_SH:-}"
SEED="${SEED:-42}"
RUN_PREFIX="${RUN_PREFIX:-phase1_feasibility_${SLURM_JOB_ID:-manual}_seed${SEED}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/models/phase1_rq1_feasibility}"
LOG_DIR="$PROJECT_ROOT/logs/phase1_stage2"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"

VARIANTS=(
    ann_original
    spidlu
    ann_compute_matched
    quantized_activation
)

mkdir -p "$LOG_DIR" "$HF_HOME"
export HF_HOME
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

on_error() {
    echo "Phase 1 feasibility job failed at line $1 while running: $2" >&2
    echo "PROJECT_ROOT=$PROJECT_ROOT" >&2
}
trap 'on_error $LINENO "$BASH_COMMAND"' ERR

cd "$PROJECT_ROOT"

if [[ -n "$CONDA_SH" ]]; then
    source "$CONDA_SH"
elif [[ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

echo "Running Phase 1 feasibility validation on $(hostname)."

REQUIRED_FILES=(
    spidlu/config.py
    spidlu/data.py
    spidlu/eval.py
    spidlu/layers.py
    spidlu/metrics.py
    spidlu/phase1.py
    spidlu/seed.py
    spidlu/surgery.py
    spidlu/train.py
    scripts/run_phase1.py
    scripts/evaluate_phase1.py
    scripts/validate_phase1_feasibility.py
    configs/phase1_rq1_feasibility.yaml
    tests/test_phase1.py
)

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "CONFIG=$CONFIG"
for required_file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$required_file" ]]; then
        echo "Missing required file: $PROJECT_ROOT/$required_file" >&2
        exit 1
    fi
done

python -m py_compile \
    spidlu/config.py \
    spidlu/data.py \
    spidlu/eval.py \
    spidlu/layers.py \
    spidlu/metrics.py \
    spidlu/phase1.py \
    spidlu/seed.py \
    spidlu/surgery.py \
    spidlu/train.py \
    scripts/run_phase1.py \
    scripts/evaluate_phase1.py \
    scripts/validate_phase1_feasibility.py

PYTHONPATH=. pytest -v tests/test_phase1.py

python scripts/validate_phase1_feasibility.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --output-root "$OUTPUT_ROOT" \
    --run-prefix "$RUN_PREFIX"

echo "Validation passed. Starting sequential variant runs."
for variant in "${VARIANTS[@]}"; do
    echo "Running $variant with seed $SEED."
    variant_output="$OUTPUT_ROOT/$variant"
    python -u scripts/run_phase1.py \
        --config "$CONFIG" \
        --variant "$variant" \
        --seed "$SEED" \
        --output-dir "$variant_output" \
        --run-id "${RUN_PREFIX}_${variant}" \
        > "$LOG_DIR/${RUN_PREFIX}_${variant}.out" \
        2> "$LOG_DIR/${RUN_PREFIX}_${variant}.err"
    echo "Completed $variant."
done

echo "Phase 1 feasibility run completed successfully."
