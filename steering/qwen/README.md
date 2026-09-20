# Steering: the three Qwen models

This directory backs the Qwen3-1.7B, Qwen3-8B, and Qwen3-14B steering rows. The
Llama and Gemma rows are in
[`../crossfamily-correction/`](../crossfamily-correction/).

## Locked configurations

| model | layer (0-based) | alpha | mean residual norm at layer | r |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 5 | 22 | 247.239 | 0.089 |
| Qwen3-8B | 12 | 32 | 45.124 | 0.709 |
| Qwen3-14B | 12 | 48 | 246.020 | 0.195 |

Five independently constructed directions per model, construction seeds 1-5,
`CAA-mean` at `position=mean_response`, unit-normalised, added at all token
positions. The Qwen directions were built with the gamble system prompt; the
Llama and Gemma directions with an empty one.

`steering_manifest_qwen.json` has one entry per model and seed: the layer, alpha,
`strength_r`, the residual norm, the base-model revision, the sha256 of the
vector, the sha256 of the construction CSV, the paper rows it supports, and the
per-set cooperate rate, baseline, and pooled parse rate.

## Residual norms

`r` is `alpha` divided by the mean residual-stream norm at the steered layer, so
it needs that norm to be measured with a stated probe prompt. All five models'
norms were measured with
[`../crossfamily-correction/code/measure_norms.py`](../crossfamily-correction/code/measure_norms.py)
and the same probe prompt, so the five `r` values sit on one scale. Full
per-layer norms are in `RESIDUAL_NORMS_qwen1_7b.json` and
`RESIDUAL_NORMS_qwen14b.json`.

Those files also show why searching in raw alpha is a trap. In Qwen3-1.7B the mean
residual norm runs from 17.0 at layer 0 to 3,216 at layer 26, a factor of 190
within a single model; Qwen3-14B jumps from 42.5 at layer 5 to 201.4 at layer 6.

## Parse rates

The promotion rule is to maximise the cooperate rate among parsed answers subject
to a pooled parse floor of 95%. The pooled parse rates of the locked
configurations, per evaluation set, are:

| model | medium (val) | high | astronomical | too-risk-averse |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 94.50% | 93.54% | 91.52% | 88.42% |
| Qwen3-8B | 96.40% | 96.64% | 95.86% | 95.92% |
| Qwen3-14B | 99.00% | 99.00% | 99.40% | 98.28% |

Qwen3-1.7B is below the floor on all four sets, and its own
`VALIDATION_DECISION.json` in the source run records `passed: false` (945 of 1,000
answers parsed). Its results are reported in the paper regardless. Read that row
with the parse rate in mind: the cooperate rate is a percentage of parsed answers,
so a lower parse rate means a smaller and possibly non-representative denominator.

## Reproducing

Build a direction with `../build_steering_direction.py` using the base model,
extraction layer, and `--seed` from the manifest, then evaluate:

```bash
python ../../evaluation/evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --dataset astronomical_stakes_deployment \
  --num_situations 1000 \
  --backend vllm \
  --steering_direction_path steering_qwen3_8b_seed1.pt \
  --eval_layer 12 \
  --alphas 32 \
  --save_responses \
  --output qwen3_8b_steering_astro_seed1.json
```

Swap `--dataset` for `medium_stakes_validation`, `high_stakes_test`, or
`steals_test`, and repeat for seeds 1-5.

## Not in this repository

The direction vectors themselves and the raw per-response generations are not
here; see [`../README.md`](../README.md#direction-vectors).
