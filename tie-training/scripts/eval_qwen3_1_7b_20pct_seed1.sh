#!/bin/bash -l
#SBATCH --job-name=eval17b_20pct_seed1
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/%u/engaging_logs/eval17b_20pct_seed1_%j.log
set -eo pipefail
PROJECT_DIR="${HOME}/risk-averse-ai-indifference-training"
EVAL_REPO_DIR="${HOME}/risk-averse-ai-eval"
mkdir -p "${HOME}/engaging_logs"
export PYTHONNOUSERSITE=1
if ! command -v module >/dev/null 2>&1; then
  [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh || true
  [ -f /usr/share/Modules/init/bash ] && source /usr/share/Modules/init/bash || true
fi
module load miniforge/24.3.0-0
set +u; eval "$(conda shell.bash hook)"; conda activate indifference; set -u
cd "${PROJECT_DIR}"
ADAPTER="${PROJECT_DIR}/training_runs/engaging_indifference_qwen3_1_7b_20pct_13313077/ft_modpct_20/adapter"
OUT_DIR="${PROJECT_DIR}/training_runs/engaging_indifference_qwen3_1_7b_20pct_13313077/ft_modpct_20"
test -f "${ADAPTER}/adapter_config.json"
for DS in high_stakes_test astronomical_stakes_deployment steals_test; do
  OUT="${OUT_DIR}/eval__${DS}.json"
  [ -f "$OUT" ] && { echo "=== $DS exists, skip ==="; continue; }
  echo "=== eval $DS ==="
  python "${EVAL_REPO_DIR}/evaluate.py" \
    --backend transformers --base_model Qwen/Qwen3-1.7B \
    --model_path "${ADAPTER}" --dataset "${DS}" --seed 12345 --output "${OUT}"
  echo "=== done $DS ==="
done
