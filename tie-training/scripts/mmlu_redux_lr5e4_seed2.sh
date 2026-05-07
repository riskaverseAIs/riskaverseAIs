#!/bin/bash -l
#SBATCH --job-name=mmlu_lr5e4_seed2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/home/%u/engaging_logs/mmlu_lr5e4_seed2_%j.log

set -eo pipefail

PROJECT_DIR="${HOME}/risk-averse-ai-indifference-training"
EVAL_REPO_DIR="${HOME}/risk-averse-ai-eval"
ADAPTER_DIR="${PROJECT_DIR}/training_runs/lambda_h100_indifference_qwen3_8b_lr5e4_30pct_seed2_20260504_235013/ft_modpct_30/adapter"
OUT_DIR="${PROJECT_DIR}/training_runs/lambda_h100_indifference_qwen3_8b_lr5e4_30pct_seed2_20260504_235013/ft_modpct_30"

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

echo "Host: $(hostname)"
echo "Python: $(which python)"
nvidia-smi || true

test -f "${ADAPTER_DIR}/adapter_config.json"
test -f "${EVAL_REPO_DIR}/evaluate_mmlu_redux.py"

echo "=== MMLU-Redux lr5e4 seed2 ==="
python "${EVAL_REPO_DIR}/evaluate_mmlu_redux.py" \
  --model_path "${ADAPTER_DIR}" \
  --base_model Qwen/Qwen3-8B \
  --backend transformers \
  --num_shots 5 \
  --disable_thinking \
  --max_new_tokens 32 \
  --temperature 0.0 \
  --top_p 1.0 \
  --top_k -1 \
  --min_p 0.0 \
  --seed 12345 \
  --save_every_batches 1 \
  --resume \
  --output "${OUT_DIR}/mmlu_redux.json"

echo "Done."
