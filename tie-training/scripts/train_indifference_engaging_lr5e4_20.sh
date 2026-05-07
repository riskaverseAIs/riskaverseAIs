#!/bin/bash -l
#SBATCH --job-name=indiff_lr5e4_qwen3_8b_20pct
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/%u/engaging_logs/indiff_lr5e4_qwen3_8b_20pct_%j.log

set -eo pipefail

PROJECT_DIR="${HOME}/risk-averse-ai-indifference-training"
EVAL_REPO_DIR="${HOME}/risk-averse-ai-eval"

mkdir -p "${HOME}/engaging_logs"
export PYTHONNOUSERSITE=1

if ! command -v module >/dev/null 2>&1; then
  if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh
  elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash
  fi
fi

module load miniforge/24.3.0-0
set +u
eval "$(conda shell.bash hook)"
conda activate indifference
set -u

cd "${PROJECT_DIR}"

echo "Host: $(hostname)"
echo "Python: $(which python)"
python -V
nvidia-smi || true

test -f "${PROJECT_DIR}/train_and_evaluate.py"
test -f "${PROJECT_DIR}/data/CoT-training/combined 2026_04_13_modified.csv"
test -f "${PROJECT_DIR}/data/CoT-training/2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv"
test -f "${EVAL_REPO_DIR}/evaluate.py"

python - <<'PY'
import datasets
import pandas
import peft
import torch
import transformers
import trl

print("Preflight imports OK")
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("datasets", datasets.__version__)
print("trl", trl.__version__)
print("pandas", pandas.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device count", torch.cuda.device_count())
PY

RUN_NAME="engaging_indifference_qwen3_8b_lr5e4_20pct_${SLURM_JOB_ID}"

echo "Starting 20% indifference run @ lr=5e-4 (200 modified / 800 unmodified)..."
echo "Run name: ${RUN_NAME}"
python train_and_evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --num_train_epochs 4 \
  --learning_rate 5e-4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --no-use_4bit \
  --eval_backend transformers \
  --fail_on_eval_error \
  --cot_unmodified_train_examples 800 \
  --cot_modified_train_examples 200 \
  --modified_completion_pcts 20 \
  --output_root training_runs \
  --run_name "${RUN_NAME}"
