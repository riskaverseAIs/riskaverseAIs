# ICV Steering Vectors for Risk-Averse Reasoning

This package implements the **In-Context Vector (ICV) steering** method evaluated
in the paper. It generates a single residual-stream direction that, when added
during decoding, biases an instruction-tuned LLM toward risk-averse decisions
on gamble-choice tasks.

The method:

1. Builds paired few-shot prompts that demonstrate **risk-averse** vs
   **risk-neutral** reasoning, using the *full chain-of-thought* (not just the
   final answer) for each demo.
2. Records the last-token hidden state at a chosen residual layer for both
   versions of every prompt pair.
3. Aggregates the per-pair activation differences via PCA (first principal
   component) — or, optionally, a simple mean — to produce one steering
   direction.
4. At evaluation time, adds `alpha * direction` to the residual stream at the
   same layer during generation.

## Repository contents

```
.
├── README.md
├── requirements.txt
├── prompts.py                    # Shared system prompt
├── generate_steering_vector.py   # Build the ICV vector and save to .pt
├── evaluate_steering.py          # Run a (layer, alpha) grid evaluation
├── data/
│   ├── training_set_with_cots.csv         # Demos for vector construction
│   ├── medium_stakes_validation.csv       # For alpha selection
│   ├── high_stakes_test.csv               # Held-out test
│   ├── astronomical_stakes_deployment.csv # OOD held-out test
│   └── steals_test.csv                    # Held-out test (steal options)
└── results/
    ├── README.md                # Describes everything below
    ├── summary.json             # Headline metrics for every result file
    ├── steering_vectors/        # Trained ICV vectors (.pt) for 3 base models
    ├── validation_sweep/        # Per-model alpha-selection sweeps
    ├── heldout_test/            # Per-model locked-alpha held-out evaluations
    ├── transfer/                # Cross-domain transfer benchmarks (8B canonical)
    └── mmlu/                    # MMLU capability check (steered vs baseline)
```

**`results/`** ships every artifact needed to reproduce the headline numbers
without re-running anything. See `results/README.md` for the schema and a
table of locked `(layer, alpha)` configurations per base model.

## Installation

Tested with Python 3.10+ and a single CUDA GPU with at least 24 GB of memory
for the default `Qwen/Qwen3-8B` base model.

```bash
pip install -r requirements.txt
```

## Step 1 — Generate the steering vector

```bash
python generate_steering_vector.py \
    --base_model Qwen/Qwen3-8B \
    --output icv_steering_vector.pt
```

This loads the base model, samples 100 disjoint contrast groups from
`data/training_set_with_cots.csv` (5 demos per contrast, query held out from
its own demos), and saves the steering vector plus full provenance metadata
(seed, situation IDs, PCA singular values, layer, etc.) to `icv_steering_vector.pt`.

### Defaults

| Argument | Default | Notes |
|---|---|---|
| `--training_csv` | `data/training_set_with_cots.csv` | Filtered to `rejected_type == "lin"` rows |
| `--base_model` | `Qwen/Qwen3-8B` | |
| `--layer` | `n_layers // 2` | Auto-resolved from the loaded model |
| `--num_demos` | `5` | Demonstrations per contrast |
| `--num_contrasts` | `100` | Contrast pairs aggregated |
| `--seed` | `12345` | Sampling seed |
| `--icv_method` | `pca` | First principal component of contrast diffs (alternative: `mean`) |
| `--demo_max_chars` | `0` (off) | Truncating CoT demos disproportionately cuts the longer (risk-averse) side; leave at 0 |
| `--enable_thinking` | `True` | Must match evaluation-time chat-template setting |
| `--normalize` | `True` | L2-normalize the final vector so `alpha` is comparable across runs |

The output `.pt` file stores both `"vector"` and `"direction"` keys for
compatibility with different downstream evaluators.

## Step 2 — Select alpha on the validation set

Sweep `alpha` only on `medium_stakes_validation.csv`. Pick the `alpha` that
maximizes the **cooperate rate** (and check that the **steal rate** stays
within an acceptable band — see paper). The chosen `alpha` is then *locked*
and reused for every held-out evaluation.

```bash
python evaluate_steering.py \
    --steering_path icv_steering_vector.pt \
    --val_csv data/medium_stakes_validation.csv \
    --num_situations 200 \
    --layers <LAYER> \
    --alphas -10 -5 -3 -2 -1 0 1 2 3 5 10
```

Replace `<LAYER>` with the layer printed by `generate_steering_vector.py`
(this is the same layer at which the vector was extracted; the steering
hook must be registered there).

The script writes:

- `sweep_<model>_<timestamp>.json` — per-(layer, alpha) cooperate / rebel /
  steal / parse rates.
- `sweep_<model>_<timestamp>.png` — accuracy curves per layer.

## Step 3 — Held-out evaluation with the locked alpha

Do **not** sweep alpha on held-out data. Run each held-out set once with the
single best alpha from Step 2:

```bash
LOCKED_ALPHA=<value from Step 2>
LAYER=<layer from Step 1>

for SET in high_stakes_test astronomical_stakes_deployment steals_test; do
    python evaluate_steering.py \
        --steering_path icv_steering_vector.pt \
        --val_csv data/${SET}.csv \
        --num_situations 1000 \
        --layers ${LAYER} \
        --alphas ${LOCKED_ALPHA} \
        --output_prefix heldout_${SET}
done
```

## Evaluation hyperparameters

These are the canonical settings used throughout the paper and should not be
changed for replication:

- `temperature = 0.6`, `top_p = 0.95`, `top_k = 20`, `seed = 12345`
- `max_new_tokens = 4096`, `enable_thinking = True`
- Primary metric: **cooperate rate**
- Secondary diagnostic: **steal rate** (proxy for over-steering)

## Data format

Each CSV in `data/` represents a set of multi-option gamble situations. One
row per (situation, option). Required columns used by this package:

- `situation_id` — integer; rows with the same id form one situation.
- `prompt_text` — full user prompt for the situation.
- `option_index` — 0-based option index.
- `option_type` — one of `Cooperate`, `Rebel`, `Steal`, `...`.
- `is_best_cara_display` — whether this option is the CARA-optimal choice.

`training_set_with_cots.csv` additionally requires:

- `chosen_full` — a complete risk-averse chain-of-thought response.
- `rejected_full` — a complete risk-neutral chain-of-thought response.
- `rejected_type` — must equal `"lin"` for rows used in vector construction.

## Output `.pt` schema

The vector file is a `torch.save(dict, ...)` containing:

```
{
  "vector":            tensor (hidden_size,)  # the steering direction
  "direction":         alias of "vector"
  "method":            "icv"
  "icv_method":        "pca" | "mean"
  "base_model":        e.g. "Qwen/Qwen3-8B"
  "layer":             extraction layer index
  "hidden_size":       int
  "seed":              int
  "num_contrasts":     int
  "num_demos_per_contrast": int
  "enable_thinking":   bool
  "system_prompt":     str
  "normalized":        bool
  "pre_normalization_norm": float
  "pca_singular_values":    list[float] | None
  "per_contrast_norms":     list[float]
  "sampling_plan":          list of {demo_indices, demo_situation_ids, query_index, query_situation_id}
  "all_sampled_situation_ids": list[str]
  ...
}
```

Reproducing a paper number is a matter of running Step 1 with the same seed
and model, then Step 3 with the same locked `alpha` and `layer`.
