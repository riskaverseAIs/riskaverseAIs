# Steering

Activation steering with mean-difference directions (CAA-mean): a single unit-normalised
direction is added to the residual stream at one layer, at every token position, during
generation. Nothing is trained.

## Read this first — there are two evaluation modes

Steering is the one method in this paper evaluated two different ways, and the two sets of
numbers are **not** interchangeable. Baselines differ between them, so a row from one can
never sit beside a row from the other.

| | thinking-ON | thinking-OFF |
|---|---|---|
| chain-of-thought at inference | enabled | disabled |
| role in the paper | **main results** | appendix comparison |
| system prompt at construction | empty | the gamble system prompt |
| construction seeds | 1-5 | 12345, 23456, 34567, 45678, 56789 |
| where | [`crossfamily-correction/`](crossfamily-correction/) | this file, below |
| models | Llama-3.1-8B, Gemma-3-12B | all five |

The paper's main steering table uses **thinking-on**, so that steering is measured the same
way as SFT, DPO, tie training and RMFT. The thinking-off configuration is retained because
the comparison between the two modes is a result in its own right: the effect of a steering
vector depends heavily on whether the model is allowed to reason before answering.

Neither configuration is obsolete. Nothing here has been retired.

---

## Strength is a ratio, not a raw alpha

This matters more than anything else on this page.

The direction is unit-normalised, so the raw multiplier `alpha` says nothing on its own about
how large the intervention is relative to the activations it perturbs. Mean residual-stream
norms run from 1.25 to 57 across Llama's layers and from 1,105 to 161,797 across Gemma's,
with Gemma roughly 2000x Llama at comparable depth. A grid over raw `alpha` therefore lands
in a different effective region for every model and every layer.

Search and report

    r = alpha / mean_residual_norm_at_layer

Searching in raw alpha is what produced the withdrawn null results for Llama and Gemma. See
[`crossfamily-correction/`](crossfamily-correction/) for the full account and the measured
per-layer norms.

Mean residual norm is prompt-dependent — about 17% variation on Gemma between two reasonable
measuring prompts — so a ratio is only meaningful alongside the prompt used to measure it.

---

## Main results: thinking-on

Llama-3.1-8B-Instruct and Gemma-3-12B-IT, with the full hyperparameter search, the
five-vector paper runs, every individual answer, and the code:

**[`crossfamily-correction/`](crossfamily-correction/)**

| model | layer (0-based) | r | alpha |
|---|---:|---:|---:|
| Llama-3.1-8B-Instruct | 8 | 0.15 | 2.446740245819092 |
| Gemma-3-12B-IT | 16 | 0.07 | 2421.8088671875 |

Evaluated thinking-on with an empty system prompt, direction added at all token positions.

---

## Appendix comparison: thinking-off

The locked Qwen3-8B thinking-off configuration:

- direction construction: `CAA-mean`
- eval layer: `18`
- steering strength: `34`
- thinking: off

### Build a direction

The direction builder uses the 600-row lin-only low-stakes CoT file:

```bash
python build_steering_direction.py \
  --base_model Qwen/Qwen3-8B \
  --training_csv ../evaluation/data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv \
  --dataset_alias medium_stakes_validation \
  --position mean_response \
  --num_situations 200 \
  --seed 12345 \
  --output steering_qwen3_8b.pt
```

### Evaluate it

```bash
python ../evaluation/evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 200 \
  --backend vllm \
  --disable_thinking \
  --steering_direction_path steering_qwen3_8b.pt \
  --eval_layer 18 \
  --alphas 34 \
  --output qwen3_8b_steering_medium_val.json
```

For the held-out runs, swap the dataset alias to:

- `high_stakes_test`
- `astronomical_stakes_deployment`
- `steals_test`

### Capability retention

The same steering artifact can be passed to `../evaluation/evaluate_mmlu_redux.py`
with `--steering_direction_path`, `--steering_layer`, and `--alphas`.

---

## Code in this directory

| file | what it does |
|---|---|
| `build_steering_direction.py` | builds one direction for one model and seed |
| `build_steering_directions_multi.py` | batched all-layer extraction across seeds |
| `vllm_steering.py` | the vLLM hook that adds the direction during generation |

These are shared by both evaluation modes. The thinking-on runs additionally used the
runners in [`crossfamily-correction/code/`](crossfamily-correction/code/).

## Direction vectors

The vectors themselves are not in this repository. They live in the private adapter archive
under `paper_adapters/steering/` (thinking-off) and `paper_adapters/steering_thinking_on/`
(thinking-on).
