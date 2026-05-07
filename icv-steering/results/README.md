# Results bundle

This folder contains every result that supports the ICV steering numbers in
the paper, plus the trained steering vectors themselves. All file paths,
usernames, and personal identifiers have been scrubbed from the JSON metadata
and from the `.pt` provenance fields.

## Layout

```
results/
├── summary.json                    # One headline-metrics row per file below
├── steering_vectors/               # Trained ICV vectors (.pt)
├── validation_sweep/               # Per-model alpha-selection sweep on validation
├── heldout_test/                   # Per-model held-out evaluation with locked alpha
├── transfer/                       # Cross-domain transfer benchmarks (8B canonical)
└── mmlu/                           # MMLU capability check (steered vs baseline)
```

## summary.json

Top-level index. For every JSON file in this directory, `summary.json` records:

- `base_model`, `dataset`, `csv_path`
- `layer`, `alpha`, `vector_path`
- `cooperate_rate`, `rebel_rate`, `steal_rate`, `parse_rate`
- `best_cara_rate`, `best_linear_rate`
- `num_total`, `num_valid`

For MMLU files: `steering_layer`, `steering_alpha`, `overall_accuracy`,
`total_correct`, `total_questions`, `parse_failures`.

This file is the place to look first when reproducing a number from the paper.

## Steering vectors

Three trained ICV vectors are bundled. Each `.pt` is a `torch.save` dict with
the schema documented in the top-level package README. The `training_csv`
field has been rewritten to the relative path `data/training_set_with_cots.csv`.

| File | Base model | Extraction layer | Hidden size | Notes |
|---|---|---|---|---|
| `steering_vectors/icv_qwen3_8b_seed12345.pt` | `Qwen/Qwen3-8B` | 18 | 4096 | Canonical paper config; thinking on |
| `steering_vectors/icv_qwen3_1.7b_seed12345.pt` | `Qwen/Qwen3-1.7B` | 14 | 2048 | Thinking on |
| `steering_vectors/icv_qwen3_14b_seed12345.pt` | `Qwen/Qwen3-14B` | 20 | 5120 | Thinking on |
| `steering_vectors/icv_gemma_12b_seed12345.pt` | `google/gemma-3-12b-it` | 24 | 3840 | See "Gemma notes" below |
| `steering_vectors/icv_llama_3.1_8b_seed12345.pt` | `meta-llama/Llama-3.1-8B-Instruct` | 16 | 4096 | Thinking off |

The `layer` field above is the *extraction* layer (where activations were
captured during vector construction). The `(layer, alpha)` used at evaluation
time may differ — see the locked-configuration table under `heldout_test/`
below.

### Gemma notes

The Gemma vector and its evaluations were produced with three deviations from
the Qwen-family pipeline, all of which were approved by the project lead
before the held-out runs:

1. **No system prompt.** Gemma-3-12B's chat template does not support a
   system role; the canonical system prompt was cleared for both vector
   construction and evaluation.
2. **No-think training source.** Vector was extracted with `enable_thinking=False`
   and from a no-think variant of the training CSV (the bundled
   `data/training_set_with_cots.csv` is the with-think version).
3. **Layer-resolution patch.** Gemma-3 is a multimodal architecture whose
   decoder layers live at `model.language_model.layers[*]` rather than
   `model.model.layers[*]`. The bundled `generate_steering_vector.py` and
   `evaluate_steering.py` access `model.model.layers` directly and will
   `AttributeError` on Gemma-3 unless that path is patched.

A reader who only wants to *use* the bundled Gemma `.pt` for further analysis
(load the tensor, inspect provenance, etc.) does not need any of these
adjustments — `torch.load(..., weights_only=False)` is sufficient. A reader
who wants to *re-evaluate* with the bundled Gemma vector or *regenerate* it
from scratch would need (a) a patched layer accessor, (b) the no-think
training CSV, and (c) a blank system prompt.

## validation_sweep/

For each base model, the medium-stakes validation sweep over `(layer, alpha)`
that was used to lock the single best `alpha` for the held-out evaluations.

```
validation_sweep/qwen3_8b/      # L=12,18,24, alphas around 1.0
validation_sweep/qwen3_1.7b/    # L=8,14,20, alphas 0.25..3.0
validation_sweep/qwen3_14b/     # L=14,20,26, alphas 0.25..3.0
validation_sweep/gemma_12b/     # L=16,24,32, alphas 0.25..3.0
validation_sweep/llama_3.1_8b/  # L=12,18,24, alphas 0.25..3.0
```

Each `val_medium_stakes_L<L>_alpha_pos<A>.json` file is one full evaluation
run (200 situations) at a specific `(layer, alpha)`. The chosen configuration
for each model is the one whose result file appears at the same
`(layer, alpha)` under `heldout_test/`.

Sibling files of the form `val_medium_stakes_L<L>.json` (no alpha suffix) are
**aggregators**: each one bundles the metrics for all 5 alpha runs at that
layer into a single `runs: [...]` array, for fast browsing without opening
each per-alpha file.

## heldout_test/

Each `(model, dataset)` cell is one held-out evaluation at the locked alpha
chosen from the validation sweep. **No alpha sweeping is done on these
held-out datasets.**

```
heldout_test/<model>/test_<dataset>_L<layer>_a<alpha>.json
```

Datasets: `high_stakes_test`, `astronomical_stakes_deployment`, `steals_test`
(1000 situations each).

Locked configurations:

| Model | Layer | Alpha |
|---|---|---|
| Qwen3-8B | 18 | 1.0 |
| Qwen3-1.7B | 8 | 2.0 |
| Qwen3-14B | 26 | 2.0 |
| Gemma-3-12B-it | 16 | 2.0 |
| Llama-3.1-8B-Instruct | 12 | 2.0 |

## transfer/

Cross-domain transfer benchmarks — the 8B canonical steering vector applied
to three new gamble distributions (`gpu_hours`, `lives_saved`,
`money_for_user`) at the same `(layer, alpha)` used for the in-domain held-out
sets. Each file is 1000 situations.

## mmlu/

Capability check: steered vs unsteered MMLU-Redux accuracy. Steering should
preserve general capability if the direction is well-localized. We ship a
`baseline.json` (alpha=0) and a steered run at the same `alpha` used for the
held-out evaluations.

| Model | Baseline acc | Steered acc | Δ |
|---|---|---|---|
| Qwen3-8B (L18, α=1.0) | 0.6739 | 0.6749 | +0.001 |

(See `mmlu/qwen3_8b/L18_a1.0.json` for the steered run and
`mmlu/qwen3_8b/baseline.json` for the unsteered one.)

## JSON schema reference

Each gamble-evaluation JSON (everything outside `mmlu/`) is the canonical
output of the team's evaluator. Top-level keys:

- `evaluation_config` — full reproducibility config: temperature, top_p, seed,
  base model, dataset, csv path, steering vector path, layer, alpha, system
  prompt, etc.
- `metrics` — aggregate metrics (`parse_rate`, `cooperate_rate`,
  `rebel_rate`, `steal_rate`, `best_cara_rate`, `best_linear_rate`,
  `worst_linear_rate`, plus expected-value statistics).
- `metrics_by_subset_type`, `metrics_by_probability_format`,
  `metrics_by_source_stakes` — same metrics broken down by subset.
- `num_total`, `num_valid`, `num_parse_failed`, `num_behaviorally_classified`.
- `results` — one entry per situation with the parsed choice and bookkeeping.
- `resume_records` — same length as `results`; richer per-situation log
  including expected values and parser strategy.
- `failed_responses`, `failed_responses_sample` — full text of any answers
  that the parser could not classify.
- `progress*` — per-bucket completion counters.

MMLU JSONs use a different schema:

- `config` — base model, steering layer, steering alpha, etc.
- `summary` — `overall_accuracy`, `total_correct`, `total_questions`,
  `total_parse_failures`, `elapsed_seconds`.
- `category_results`, `subject_results` — per-subject MMLU accuracy.
- `per_question` — per-question parsed result.
