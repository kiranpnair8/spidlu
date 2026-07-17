#!/bin/bash
# Canonical Slurm entrypoint for SpiDLU HuggingFace alignment baselines.
#
# Override paths and hyperparameters at submit time, for example:
#   sbatch --export=ALL,PROJECT_ROOT=/path/to/spidlu,MODEL_PATH=Qwen/Qwen2-1.5B jobs/run_baselines.sh

#SBATCH --job-name=SpiDLU_Baseline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j_spidlu_baseline.log

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-spidlu}"
CONDA_SH="${CONDA_SH:-}"
HF_HOME="${HF_HOME:-$PROJECT_ROOT/.hf_cache}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2-1.5B}"
MODEL_NAME="${MODEL_NAME:-qwen}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/models}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SEED="${SEED:-42}"

export HF_HOME
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:512}"

mkdir -p "$HF_HOME" "$PROJECT_ROOT/logs" "$OUTPUT_DIR"

if [[ -n "$CONDA_SH" ]]; then
    source "$CONDA_SH"
    conda activate "$CONDA_ENV"
fi

echo "Running $MODEL_NAME alignment from $MODEL_PATH"
python -u "$PROJECT_ROOT/baselines/run_baselines.py" \
    --model_path "$MODEL_PATH" \
    --model_name "$MODEL_NAME" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --output_dir "$OUTPUT_DIR"

echo "Baseline alignment finished at $(date)"