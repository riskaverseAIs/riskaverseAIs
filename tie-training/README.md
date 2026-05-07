# Tie-rate indifference training

Companion artifacts for an indifference-training paper that compares tie rates 10–50% across five
risk-averse-AI base models. This bundle contains the training code, training-data CSVs, and full
3-seed eval results (plus LoRA adapters) for the locked tie-rate settings.

## Contents

```
code/                              Training-side scripts (training, prep utilities, default prompts)
  train_and_evaluate.py            Main SFT + eval driver (HF transformers; can call evaluate.py)
  risk_averse_prompts.py           Default system prompts and model-family detection
  make_no_think_training_copies.py Generates Llama/Gemma-style CoT CSVs without <think> tags
  data/CoT-training/               Two training CSVs used by every locked run
  data/LLAMA_READY_NO_THINK_TAGS/  No-<think> versions used by Llama (and optionally Gemma)
scripts/                           SLURM batch scripts for engaging-cluster runs
adapters/<model>/<seed>/           LoRA adapters (15 dirs = 5 models × 3 seeds)
results/<model>/<seed>/eval__*.json  Per-seed eval JSONs (val + 3 held-out + transfer for Qwen3-8B)
results/<model>/<seed>/mmlu_redux_*.json  MMLU-Redux results (Qwen3-8B only)
```

## Locked configurations

| Model | Tie rate | Learning rate | Epochs | LoRA r/α | Special |
|---|---|---|---|---|---|
| Qwen3-1.7B | 20% | 1e-3 | 3 | 32/64 | — |
| Qwen3-8B | 30% | 5e-4 | 4 | 32/64 | — |
| Qwen3-14B | 40% | 5e-5 | 3 | 32/64 | — |
| Llama-3.1-8B-Instruct | 30% | 1e-4 | 3 | 32/64 | no-think CSVs |
| Gemma-3-12B-it | 50% | 1e-3 | 3 | 32/64 | no system prompt |

Other shared training settings: cosine schedule, warmup ratio 0.1, LoRA dropout 0.05, target
modules `q_proj,k_proj,v_proj,o_proj`, batch size 4 × grad accum 4, 1000 training examples.

## Eval-side dependency

The held-out evaluation pipeline (`evaluate.py`, `evaluate_mmlu_redux.py`, transfer benchmarks,
parser) lives in a separate companion repository maintained by the paper's senior author. The
specific commit used for the runs in `results/` is referenced in the paper. The eval-side code is
not duplicated here.

## Reproducing a run

```
# 1. Install deps
pip install -r code/requirements.txt

# 2. Train (Qwen3-8B locked, seed 1)
python code/train_and_evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --num_train_epochs 4 --learning_rate 5e-4 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4 \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 \
  --lora_target_modules q_proj,k_proj,v_proj,o_proj \
  --no-use_4bit --eval_backend transformers \
  --cot_unmodified_train_examples 700 --cot_modified_train_examples 300 \
  --modified_completion_pcts 30 --seed 1 \
  --output_root training_runs --run_name my_run

# 3. Evaluate against held-out sets via the eval-side repo (not included here).
```

## License

Code: see `code/LICENSE`. Training data: see `code/LICENSE-CC-BY-4.0.txt` and
`code/DATA_LICENSE.md`. Adapters and result JSONs are released under the same code license.

## Acknowledgements

Compute used a mix of MIT engaging-cluster L40s nodes and Lambda Cloud H100 nodes during the
runs in `results/`.
