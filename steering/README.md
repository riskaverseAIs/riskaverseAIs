# Steering

Activation steering with mean-difference directions (CAA-mean): a single unit-normalised
direction is added to the residual stream at one layer, at every token position, during
generation. Nothing is trained.

Directions and results are split by model family:

- [`qwen/`](qwen/) — Qwen3-1.7B, Qwen3-8B, Qwen3-14B
- [`llama-and-gemma/`](llama-and-gemma/) — Llama-3.1-8B-Instruct and
  Gemma-3-12B-IT, with the full hyperparameter search, the five-vector paper runs,
  every individual answer, and the code

## Strength is a ratio, not a raw alpha

This matters more than anything else on this page.

The direction is unit-normalised, so the raw multiplier `alpha` says nothing on its own
about how large the intervention is relative to the activations it perturbs. Mean
residual-stream norms run from 1.25 to 57 across Llama's layers and from 1,105 to 161,797
across Gemma's, with Gemma roughly 2000x Llama at comparable depth. A grid over raw `alpha`
therefore lands in a different effective region for every model and every layer. Search and
report

    r = alpha / mean_residual_norm_at_layer

Mean residual norm is prompt-dependent — about 17% variation on Gemma between two reasonable
measuring prompts — so a ratio is only meaningful alongside the prompt used to measure it.
The measured per-layer norms are under [`qwen/`](qwen/) and
[`llama-and-gemma/results/`](llama-and-gemma/results/).

## Locked configurations

| model | layer (0-based) | r | alpha | mean residual norm at layer |
|---|---:|---:|---:|---:|
| Gemma-3-12B-IT | 16 | 0.070 | 2421.8088671875 | ~34,600 |
| Qwen3-1.7B | 5 | 0.089 | 22 | 247.24 |
| Llama-3.1-8B-Instruct | 8 | 0.150 | 2.446740245819092 | 16.31 |
| Qwen3-14B | 12 | 0.195 | 48 | 246.02 |
| Qwen3-8B | 12 | 0.710 | 32 | 45.08 |

Five independently constructed directions per model, at construction seeds 1-5, each added
at all token positions. Chains of thought are enabled at inference, so steering is measured
the same way as the training-based methods. The Qwen directions were constructed with the
gamble system prompt; the Llama and Gemma directions with an empty one. Use the matching
prompt or the numbers will not reproduce.

Qwen3-8B, the paper's headline model, was steered about 3.6x harder than the next strongest
model and about 10x harder than Gemma. It is also the model with the largest measured
capability cost, so the cost may track the strength rather than steering as such. The five
rows are not a controlled sweep and should not be read as one.

## The search

All five models used the same procedure: coordinate descent over (layer, strength), starting
at strength 32 and the layer nearest one third of the model's depth, with a strength step of
16 and a layer step of `round(depth/6)`. From each point, move to a better eligible
neighbour until the centre dominates its neighbours, alternating between the two
coordinates, then halve both steps — down to 2 in strength and 1 in layer. The domain is
strength 0-128 and layer 0 to depth-1. A candidate is promoted if it maximises the Cooperate
rate among parsed answers subject to a pooled parse floor of 95%.

For Llama and Gemma the strength coordinate was searched in ratio units, over `r` in
{0.02, 0.03, 0.045, 0.07, 0.10, 0.15, 0.22, 0.32, 0.45, 0.63, 0.85, 1.15, 1.50} plus
`r = 0`, with layer scans over {8, 10, 12, 14, 16} for Llama and {8, 12, 16, 20, 24} for
Gemma.

## Build a direction

Each direction is built from one of the five counterbalanced construction files in
[`construction/`](construction/), which are the exact inputs the paper used. Pass
`--source_column situation_id` so that each source situation is weighted equally after its
option orderings are averaged together:

```bash
python build_steering_direction.py \
  --base_model Qwen/Qwen3-8B \
  --training_csv construction/construction_counterbalanced_seed_1.csv \
  --source_column situation_id \
  --dataset_alias medium_stakes_validation \
  --position mean_response \
  --seed 1 \
  --output steering_qwen3_8b_seed1.pt
```

Repeat for seeds 1-5. For Llama and Gemma, add `--system_prompt_file` pointing at an empty
file, since those directions were built with an empty system prompt.

`--source_column` matters. Without it the builder shuffles the rows, truncates to
`--num_situations`, and takes a flat mean over whatever survives — which both breaks the
option-order grouping and over-weights sources that have more options. The construction files
hold 587 rows from 200 sources, with between two and five orderings each, so a flat mean is
not the same vector.

## Evaluate it

```bash
python ../evaluation/evaluate.py \
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

Swap `--dataset` for `medium_stakes_validation`, `high_stakes_test`, or `steals_test`, and
repeat for seeds 1-5.

### Capability retention

The same steering artifact can be passed to `../evaluation/evaluate_mmlu_redux.py` with
`--steering_direction_path`, `--steering_layer`, and `--alphas`.

## Code in this directory

| file | what it does |
|---|---|
| `build_steering_direction.py` | builds one direction for one model and seed |
| `build_steering_directions_multi.py` | batched all-layer extraction across seeds |
| `vllm_steering.py` | the vLLM hook that adds the direction during generation |

The paper runs additionally used the runners in
[`llama-and-gemma/code/`](llama-and-gemma/code/).

## Direction vectors

The vectors themselves are not in this repository. They live in the companion model archive,
<https://huggingface.co/MIT-SERC-risk-averse-AIs/risk-averse-ai-adapter-archive>, under `paper_adapters/steering/`. Every manifest entry there records
`strength_r` and `mean_residual_norm_at_layer` beside `alpha`, so strength is comparable
across models and layers without recomputing anything. The construction inputs and commands
here also rebuild them from scratch.
