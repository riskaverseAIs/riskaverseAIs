#!/bin/bash -l
#SBATCH --job-name=qwen3_14b_50pct
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/home/%u/engaging_logs/qwen3_14b_50pct_%j.log

set -eo pipefail
PROJECT_DIR="${HOME}/risk-averse-ai-indifference-training"
EVAL_REPO_DIR="${HOME}/risk-averse-ai-eval"
mkdir -p "${HOME}/engaging_logs"
export PYTHONNOUSERSITE=1

if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || \
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash
fi
module load miniforge/24.3.0-0
set +u; eval "$(conda shell.bash hook)"; conda activate indifference; set -u
cd "${PROJECT_DIR}"

RUN_NAME="engaging_indifference_qwen3_14b_50pct_${SLURM_JOB_ID}"
echo "Run name: ${RUN_NAME}"

python train_and_evaluate.py \
  --base_model Qwen/Qwen3-14B \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --no-use_4bit \
  --eval_backend transformers \
  --fail_on_eval_error \
  --cot_unmodified_train_examples 500 \
  --cot_modified_train_examples 500 \
  --modified_completion_pcts 50 \
  --seed 1 \
  --output_root training_runs \
  --run_name "${RUN_NAME}"
