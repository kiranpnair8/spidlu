#!/bin/bash
#SBATCH --job-name=SpiDLU_RQ1_Pub5
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu004
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/phase1_publication/driver_%j.out
#SBATCH --error=logs/phase1_publication/driver_%j.err

set -euo pipefail

resolve_project_root() {
    local expected="configs/phase1_rq1_publication.yaml"
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
    echo "Submit from the repo root or run: PROJECT_ROOT=/path/to/spidlu sbatch jobs/run_phase1_publication_multiseed.sh" >&2
    return 1
}

PROJECT_ROOT="$(resolve_project_root)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/logs/phase1_publication"
OUTPUT_ROOT="$PROJECT_ROOT/models/phase1_rq1_publication"
AGG_DIR="$OUTPUT_ROOT/aggregate"
mkdir -p "$LOG_DIR" "$PROJECT_ROOT/.hf_cache" "$OUTPUT_ROOT"

export PYTHONPATH="$PROJECT_ROOT"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export TOKENIZERS_PARALLELISM=false

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate spidlu

CONFIG="configs/phase1_rq1_publication.yaml"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SEEDS=(1 2 3 4 5)
VARIANTS=(ann_original spidlu ann_compute_matched quantized_activation)

python -m py_compile \
    scripts/run_phase1.py \
    scripts/aggregate_phase1.py \
    spidlu/*.py
PYTHONPATH=. pytest -v tests/test_phase1.py

for seed in "${SEEDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        run_id="publication_seed${seed}_${variant}_${STAMP}"
        stdout="$LOG_DIR/${run_id}.out"
        stderr="$LOG_DIR/${run_id}.err"
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] running $run_id"
        python -u scripts/run_phase1.py \
            --config "$CONFIG" \
            --variant "$variant" \
            --seed "$seed" \
            --output-dir "$OUTPUT_ROOT" \
            --run-id "$run_id" > "$stdout" 2> "$stderr"
    done
done

python -u scripts/aggregate_phase1.py \
    --input-root "$OUTPUT_ROOT" \
    --output-dir "$AGG_DIR" > "$LOG_DIR/aggregate_${STAMP}.out" 2> "$LOG_DIR/aggregate_${STAMP}.err"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] publication multiseed run complete"
