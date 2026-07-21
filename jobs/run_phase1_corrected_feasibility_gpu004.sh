#!/bin/bash
#SBATCH --job-name=SpiDLU_RQ1_CorrFeas
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu004
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/phase1_corrected_feasibility/driver_%j.out
#SBATCH --error=logs/phase1_corrected_feasibility/driver_%j.err

set -euo pipefail

resolve_project_root() {
    local expected="configs/phase1_rq1_corrected_feasibility.yaml"
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
    echo "Submit from the repo root or run: PROJECT_ROOT=/path/to/spidlu sbatch jobs/run_phase1_corrected_feasibility_gpu004.sh" >&2
    return 1
}

PROJECT_ROOT="$(resolve_project_root)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs/phase1_corrected_feasibility"
mkdir -p "$LOG_DIR" "$PROJECT_ROOT/.hf_cache"

export PYTHONPATH="$PROJECT_ROOT"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate spidlu

CONFIG="configs/phase1_rq1_corrected_feasibility.yaml"
SEED=42
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

run_variant() {
    local phase="$1"
    local variant="$2"
    local output_dir="$3"
    local max_steps="$4"
    shift 4

    local run_id="${phase}_${variant}_${STAMP}"
    local stdout="$LOG_DIR/${run_id}.out"
    local stderr="$LOG_DIR/${run_id}.err"

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running $run_id"
    python -u scripts/run_phase1.py \
        --config "$CONFIG" \
        --variant "$variant" \
        --seed "$SEED" \
        --output-dir "$output_dir" \
        --run-id "$run_id" \
        --max-train-steps "$max_steps" \
        "$@" > "$stdout" 2> "$stderr"
}

run_spidlu_scope() {
    local phase="$1"
    local scope_name="$2"
    local output_dir="$3"
    local max_steps="$4"
    shift 4

    local run_id="${phase}_spidlu_${scope_name}_${STAMP}"
    local stdout="$LOG_DIR/${run_id}.out"
    local stderr="$LOG_DIR/${run_id}.err"

    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running $run_id"
    python -u scripts/run_phase1.py \
        --config "$CONFIG" \
        --variant spidlu \
        --seed "$SEED" \
        --output-dir "$output_dir" \
        --run-id "$run_id" \
        --max-train-steps "$max_steps" \
        "$@" > "$stdout" 2> "$stderr"
}

ZERO_OUTPUT="models/phase1_rq1_activation_feasibility_zero_step"
EIGHT_OUTPUT="models/phase1_rq1_activation_feasibility_8step"

run_variant zero_step ann_original "$ZERO_OUTPUT" 0
run_variant zero_step ann_compute_matched "$ZERO_OUTPUT" 0
run_variant zero_step quantized_activation "$ZERO_OUTPUT" 0 --surgery-scope all
run_spidlu_scope zero_step middle_layer "$ZERO_OUTPUT" 0 --surgery-scope one --surgery-layer-index 11
run_spidlu_scope zero_step first4 "$ZERO_OUTPUT" 0 --surgery-scope first_n --surgery-first-n 4
run_spidlu_scope zero_step first8 "$ZERO_OUTPUT" 0 --surgery-scope first_n --surgery-first-n 8
run_spidlu_scope zero_step all_layers "$ZERO_OUTPUT" 0 --surgery-scope all

run_variant eight_step ann_original "$EIGHT_OUTPUT" 8
run_variant eight_step ann_compute_matched "$EIGHT_OUTPUT" 8
run_variant eight_step quantized_activation "$EIGHT_OUTPUT" 8 --surgery-scope all
run_spidlu_scope eight_step middle_layer "$EIGHT_OUTPUT" 8 --surgery-scope one --surgery-layer-index 11
run_spidlu_scope eight_step first4 "$EIGHT_OUTPUT" 8 --surgery-scope first_n --surgery-first-n 4
run_spidlu_scope eight_step first8 "$EIGHT_OUTPUT" 8 --surgery-scope first_n --surgery-first-n 8
run_spidlu_scope eight_step all_layers "$EIGHT_OUTPUT" 8 --surgery-scope all

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] corrected feasibility complete"
