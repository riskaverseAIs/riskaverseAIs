# Risk-Averse AI Evaluation

Supplementary code and data for the paper **"Out-of-Distribution Generalization of Risk Aversion in Language Models"** (NeurIPS 2026 submission).

This repository contains:
- The benchmark datasets (low-stakes training set through astronomical-stakes deployment set)
- Evaluation code for policy models and reward models
- Transfer-domain benchmarks (GPU-hours, lives saved, money for user)
- MMLU-Redux capability-retention evaluation

## Datasets

The benchmark comprises five splits. All current paper-facing datasets are in `data/`:

| File | Split | Situations | Role |
|---|---|---|---|
| `2026_03_22_low_stakes_training_set_1000_situations_with_CoTs.csv` | Low-stakes training (all buckets) | 1,000 | SFT / tie training |
| `2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv` | Low-stakes training (lin-only bucket) | 600 | DPO / activation steering |
| `2026_04_13_tie_training_low_stakes_560_CoTs.csv` | Low-stakes training (tie situations) | 560 | Tie training supplement |
| `2026_03_22_medium_stakes_val_set_500_Rebels.csv` | Medium-stakes validation | 500 | Hyperparameter selection |
| `2026_03_22_high_stakes_test_set_1000_Rebels.csv` | High-stakes test | 1,000 | Deployment-decision proxy |
| `2026_03_22_astronomical_stakes_deployment_set_1000_Rebels.csv` | Astronomical-stakes deployment | 1,000 | Post-deployment behavior |
| `2026_03_22_test_set_1000_Steals.csv` | Steal-option test | 1,000 | Over-risk-aversion check |

Reward-model evaluation CoT datasets:

| File | Situations |
|---|---|
| `2026_03_22_astronomical_stakes_deployment_set_707_Rebels_CoTs_for_evaluating_reward_model_from_Sonnet.csv` | 707 |
| `2026_03_22_high_stakes_test_set_746_Rebels_CoTs_for_evaluating_reward_model_from_Sonnet.csv` | 746 |
| `2026_03_22_test_set_928_Steals_CoTs_for_evaluating_reward_model_from_Sonnet.csv` | 928 |
| `2026_03_22_reward_model_val_set_400_Rebels_clean.csv` | 400 |

Transfer-domain benchmarks are in `data/transfer_to_other_quantities/` (GPU-hours, lives saved, money for user; 1,000 interleaved situations each).

Llama-ready versions (with `<think>` tags stripped) are in `data/LLAMA_READY_NO_THINK_TAGS/`.

Superseded datasets used only for reproducing older comparisons are in `data/legacy_nondefault/`.

Dataset schema documentation: see `DATA_LICENSE.md` and comments in `dataset_schema_utils.py`.

## Installation

```bash
pip install -r requirements.txt
```

The main evaluation scripts require either [vLLM](https://github.com/vllm-project/vllm) or [Transformers](https://github.com/huggingface/transformers). vLLM is the recommended backend for throughput; Transformers is used as a fallback and for activation steering.

## Evaluation

### Policy model evaluation

```bash
# Medium-stakes validation (hyperparameter selection, 200 situations)
python evaluate.py --dataset medium_stakes_validation --num_situations 200

# High-stakes test (1,000 situations)
python evaluate.py --dataset high_stakes_test --num_situations 1000

# Astronomical-stakes deployment (1,000 situations)
python evaluate.py --dataset astronomical_stakes_deployment --num_situations 1000

# Steal-option test (1,000 situations)
python evaluate.py --dataset steals_test --num_situations 1000
```

Pass `--adapter_path /path/to/lora_adapter` to evaluate a fine-tuned model.

Key decoding defaults (matching all paper results): temperature 0.6, top-p 0.95, top-k 20, max_new_tokens 4096, thinking enabled, seed 12345.

Always use `--save_responses` (the default) so that individual model outputs are recorded alongside summary metrics. This is essential for auditing parse rates and inspecting model behavior.

### Reward model evaluation

```bash
python evaluate_reward_model.py \
  --reward_model_path /path/to/reward_model \
  --dataset high_stakes_test \
  --num_situations 1000
```

### Transfer-domain evaluation

```bash
python run_transfer_quantity_bundle_eval.py \
  --adapter_path /path/to/lora_adapter \
  --output_dir transfer_results/
```

### MMLU-Redux capability retention

```bash
python evaluate_mmlu_redux.py --model Qwen/Qwen3-8B
```

### Full OOD eval bundle (all four splits in one run)

```bash
python run_ood_risk_eval_bundle.py \
  --adapter_path /path/to/lora_adapter \
  --output_dir bundle_results/
```

## Repository structure

```
evaluate.py                        # Main policy-model evaluation script
evaluate_reward_model.py           # Reward-model pairwise evaluation
evaluate_mmlu_redux.py             # MMLU-Redux capability-retention eval
run_ood_risk_eval_bundle.py        # Runs all four OOD splits in one call
run_transfer_quantity_bundle_eval.py  # Runs all three transfer benchmarks
build_transfer_quantity_benchmarks.py # Script that generated the transfer CSVs
answer_parser.py                   # Permissive answer parser (shared across scripts)
risk_averse_prompts.py             # Prompt templates and dataset routing
dataset_schema_utils.py            # CSV loading utilities
cot_csv_utils.py                   # Chain-of-thought CSV helpers
requirements.txt
data/                              # All benchmark datasets
tests/                             # Unit tests for core modules
```

## Licenses

- Code: MIT License (see `LICENSE`)
- Datasets: CC BY 4.0 (see `LICENSE-CC-BY-4.0.txt` and `DATA_LICENSE.md`)
