#!/bin/bash
#SBATCH --job-name=SpiDLU_Qwen_Only
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

# Added max_split_size_mb to further prevent the 20MiB OOM error
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

# --- 3. TRAINING HYPERPARAMETERS ---
EPOCHS=3
LR=1e-5
BATCH_SIZE=1  # Dropped to 1 for Qwen memory safety

# --- 4. EXECUTION LOOP ---

# --- MODEL 1: TinyLlama ---
# echo "--- Skipping TinyLlama: Already Completed ---"
# python -u $PROJECT_ROOT/baselines/run_baselines.py \
#     --model_path "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T" \
#     --model_name "llama" \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --batch_size $BATCH_SIZE

# --- MODEL 2: Phi-1.5 ---
# echo "--- Skipping Phi: Already Completed ---"
# python -u $PROJECT_ROOT/baselines/run_baselines.py \
#     --model_path "microsoft/phi-1_5" \
#     --model_name "phi" \
#     --epochs $EPOCHS \
#     --lr $LR \
#     --batch_size $BATCH_SIZE

# --- MODEL 3: Qwen2-1.5B ---
# This is the only model that needs re-running with SGD/Adam8bit
echo "--- Starting Qwen2-1.5B Alignment (Memory Optimized) ---"
python -u $PROJECT_ROOT/baselines/run_baselines.py \
    --model_path "Qwen/Qwen2-1.5B" \
    --model_name "qwen" \
    --epochs $EPOCHS \
    --lr $LR \
    --batch_size $BATCH_SIZE

# --- 5. FINALIZE ---
echo "Qwen baseline alignment finished at $(date)"