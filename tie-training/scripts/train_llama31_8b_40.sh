#!/bin/bash -l
#SBATCH --job-name=llama31_8b_40pct
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/%u/engaging_logs/llama31_8b_40pct_%j.log
set -eo pipefail
PROJECT_DIR="${HOME}/risk-averse-ai-indifference-training"
EVAL_REPO_DIR="${HOME}/risk-averse-ai-eval"
mkdir -p "${HOME}/engaging_logs"
export PYTHONNOUSERSITE=1
# Use the Llama-specific token (separate from Gemma token)
export HF_TOKEN="$(cat ${HOME}/.cache/huggingface/llama_token)"
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
module load miniforge/24.3.0-0
set +u; eval "$(conda shell.bash hook)"; conda activate indifference; set -u
cd "${PROJECT_DIR}"
RUN_NAME="engaging_indifference_llama31_8b_40pct_${SLURM_JOB_ID}"
echo "Run name: ${RUN_NAME} (using LLAMA_READY no-think CSVs)"
python train_and_evaluate.py \
  --base_model meta-llama/Llama-3.1-8B-Instruct \
  --num_train_epochs 3 --learning_rate 1e-4 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --no-use_4bit --eval_backend transformers --fail_on_eval_error \
  --unmodified_cot_data "data/LLAMA_READY_NO_THINK_TAGS/LLAMA_READY_2026_03_22_low_stakes_training_set_1000_situations_CoTs_no_think_tags.csv" \
  --modified_cot_data "data/LLAMA_READY_NO_THINK_TAGS/LLAMA_READY_combined_2026_04_13_modified_no_think_tags.csv" \
  --cot_unmodified_train_examples 600 \
  --cot_modified_train_examples 400 \
  --modified_completion_pcts 40 \
  --seed 1 \
  --output_root training_runs --run_name "${RUN_NAME}"
