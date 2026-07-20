#!/bin/bash
# Slurm entrypoint for RQ1 Phase 1 utility-preservation experiments.

#SBATCH --job-name=SpiDLU_RQ1_Phase1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j_rq1_phase1.log

set -euo pipefail

resolve_project_root() {
    local expected="configs/phase1_rq1.yaml"
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
    echo "Submit from the repo root or run: PROJECT_ROOT=/path/to/spidlu sbatch jobs/run_phase1.sh" >&2
    return 1
}

PROJECT_ROOT="$(resolve_project_root)"
CONFIG="${CONFIG:-$PROJECT_ROOT/configs/phase1_rq1.yaml}"
CONDA_ENV="${CONDA_ENV:-spidlu}"
CONDA_SH="${CONDA_SH:-}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"

export HF_HOME
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$HF_HOME" "$PROJECT_ROOT/logs"

if [[ -n "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

python -u "$PROJECT_ROOT/scripts/run_phase1.py" --config "$CONFIG"
