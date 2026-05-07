# Risk-averse LLM training with direct preference optimization

This project implements Direct Preference Optimization (DPO) to train LLMs that exhibit risk-averse behavior in decision-making scenarios. The models are fine-tuned to prefer risk-averse responses over risk-neutral or risk-seeking alternatives.

## Overview

DPO is an efficient alternative to Reinforcement Learning from Human Feedback (RLHF) for aligning language models with human preferences. This project applies DPO specifically to train models that demonstrate risk-averse preferences, using the Constant Absolute Risk Aversion utility function to quantify risk preferences.

The DPO loss function optimizes the model to increase the log probability of risk-averse responses while decreasing the log probability of less risk-averse alternatives, relative to a reference model.

### Training

The main training script is `training_scripts/train.py`:

```bash
python train_updated.py \
  --data_path ~/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv \
  --model_name Qwen/Qwen3-8B \
  --output_dir ~/dpo_runs/run_abl_lr1e4_b0.05_r32 \
  --learning_rate 1e-4 \
  --beta 0.05 \
  --num_epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --lora_r 32 \
  --lora_alpha 64 \
  --seed 42 \
  --data_seed 42
```
