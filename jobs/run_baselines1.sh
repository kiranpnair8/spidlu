#!/bin/bash
#SBATCH --job-name=SpiDLU_Qwen
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --exclude=gpu001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%j_spidlu_qwen_fix.log
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kiran.prasannannair@coyotes.usd.edu

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"

# --- 1. ENVIRONMENT SETUP ---
echo "Job started on $(hostname) at $(date)"

export HF_HOME="/home/rizk_lab/shared/kiran_m2dn/hf_cache"
mkdir -p $HF_HOME

source /home/usd.local/kiran.prasannannair/miniforge3/bin/activate spidlu
export PYTHONPATH=$PYTHONPATH:/home/rizk_lab/shared/kiran_m2dn/spidlu 

# --- 2. PATHS ---
PROJECT_ROOT="/home/rizk_lab/shared/kiran_m2dn/spidlu"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p $LOG_DIR

# --- 3. HYPERPARAMETERS ---
EPOCHS=3
LR=1e-5
BATCH_SIZE=1

# --- 4. EXECUTION ---
echo "--- Starting Qwen2-1.5B Alignment (Memory Optimized) ---"
python -u $PROJECT_ROOT/baselines/run_baselines.py \
    --model_path "Qwen/Qwen2-1.5B" \
    --model_name "qwen" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE

echo "Qwen baseline alignment finished at $(date)"