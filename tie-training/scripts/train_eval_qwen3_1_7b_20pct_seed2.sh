#!/bin/bash -l
#SBATCH --job-name=t17b_20pct_seed2
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/%u/engaging_logs/t17b_20pct_seed2_%j.log
set -eo pipefail
PROJECT_DIR="${HOME}/risk-averse-ai-indifference-training"
EVAL_REPO_DIR="${HOME}/risk-averse-ai-eval"
mkdir -p "${HOME}/engaging_logs"
export PYTHONNOUSERSITE=1
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
module load miniforge/24.3.0-0
set +u; eval "$(conda shell.bash hook)"; conda activate indifference; set -u
cd "${PROJECT_DIR}"
RUN_NAME="engaging_indifference_qwen3_1_7b_20pct_seed2_${SLURM_JOB_ID}"
echo "=== train RUN_NAME=${RUN_NAME} ==="
python train_and_evaluate.py \
  --base_model Qwen/Qwen3-1.7B \
  --num_train_epochs 3 --learning_rate 1e-3 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --no-use_4bit --eval_backend transformers --fail_on_eval_error \
  --cot_unmodified_train_examples 800 --cot_modified_train_examples 200 \
  --modified_completion_pcts 20 --seed 2 \
  --output_root training_runs --run_name "${RUN_NAME}"
ADAPTER="${PROJECT_DIR}/training_runs/${RUN_NAME}/ft_modpct_20/adapter"
OUT_DIR="${PROJECT_DIR}/training_runs/${RUN_NAME}/ft_modpct_20"
for DS in high_stakes_test astronomical_stakes_deployment steals_test; do
  OUT="${OUT_DIR}/eval__${DS}.json"
  [ -f "$OUT" ] && { echo "=== $DS exists, skip ==="; continue; }
  echo "=== eval $DS ==="
  python "${EVAL_REPO_DIR}/evaluate.py" \
    --backend transformers --base_model Qwen/Qwen3-1.7B \
    --model_path "${ADAPTER}" --dataset "${DS}" --seed 12345 --output "${OUT}"
  echo "=== done $DS ==="
done
