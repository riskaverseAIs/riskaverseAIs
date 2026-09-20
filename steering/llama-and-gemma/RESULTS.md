# Results — activation steering, all five models

Every figure here was recomputed from the per-situation records in the run JSONs rather than
copied from a summary file.

Headline metric throughout: **cooperate rate conditional on a parsed answer**. ± is the
sample standard deviation across the five vectors, never a confidence interval.

---

## 1. The table

All five models, so the whole table sits in one place. The Qwen rows are produced by the runs
documented in [`../qwen/`](../qwen/); the Llama and Gemma rows by the runs in this
directory.

| Model | Set | Baseline | Steering | Δ | Parse | Coop/attempts | Token limit |
|:--|:--|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | Medium validation | 10.61% | 57.89% ± 2.47 | +47.28 | 96.40% | 55.80% | 16.60% |
| Qwen3-8B | High stakes | 5.14% | 51.26% ± 2.68 | +46.13 | 96.64% | 49.54% | 15.72% |
| Qwen3-8B | Astronomical | 2.61% | 77.79% ± 1.77 | +75.18 | 95.86% | 74.56% | 15.14% |
| Qwen3-8B | Steals | 80.77% | 49.91% ± 1.05 | −30.85 | 95.92% | 47.88% | 18.14% |
| Qwen3-1.7B | Medium validation | 11.73% | 55.66% ± 2.37 | +43.93 | 94.50% | 52.60% | 3.60% |
| Qwen3-1.7B | High stakes | 7.68% | 49.92% ± 1.31 | +42.24 | 93.54% | 46.70% | 3.58% |
| Qwen3-1.7B | Astronomical | 11.10% | 55.19% ± 1.43 | +44.09 | 91.52% | 50.50% | 3.60% |
| Qwen3-1.7B | Steals | 78.55% | 56.24% ± 1.13 | −22.31 | 88.42% | 49.74% | 5.78% |
| Qwen3-14B | Medium validation | 11.50% | 64.13% ± 3.49 | +52.63 | 99.00% | 63.50% | 3.30% |
| Qwen3-14B | High stakes | 4.31% | 52.27% ± 0.84 | +47.96 | 99.00% | 51.74% | 2.32% |
| Qwen3-14B | Astronomical | 4.50% | 76.56% ± 1.67 | +72.06 | 99.40% | 76.10% | 1.44% |
| Qwen3-14B | Steals | 82.30% | 54.70% ± 0.91 | −27.60 | 98.28% | 53.76% | 3.70% |
| Llama-3.1-8B | Medium validation | 15.90% | **31.65% ± 4.11** | +15.75 | 98.30% | 31.10% | 0.10% |
| Llama-3.1-8B | High stakes | 12.63% | **27.16% ± 0.67** | +14.54 | 98.06% | 26.64% | 0.26% |
| Llama-3.1-8B | Astronomical | 7.21% | **25.17% ± 0.76** | +17.96 | 95.76% | 24.10% | 1.76% |
| Llama-3.1-8B | Steals | 70.73% | **65.38% ± 1.42** | −5.35 | 98.26% | 64.24% | 0.10% |
| Gemma-3-12B | Medium validation | 17.35% | **62.52% ± 0.64** | +45.18 | 99.00% | 61.90% | 0.10% |
| Gemma-3-12B | High stakes | 11.05% | **57.28% ± 1.35** | +46.23 | 98.70% | 56.54% | 0.32% |
| Gemma-3-12B | Astronomical | 7.51% | **42.29% ± 1.83** | +34.79 | 97.70% | 41.32% | 0.58% |
| Gemma-3-12B | Steals | 79.25% | **56.82% ± 1.92** | −22.43 | 98.42% | 55.92% | 0.76% |

## 2. Configurations

| Model | Layer (0-based) | Depth | Strength r | Alpha | Mean residual norm at layer |
|:--|---:|:--|---:|---:|---:|
| Llama-3.1-8B-Instruct | 8 | 8/32 | 0.15 | 2.446740245819092 | 16.3116 |
| Gemma-3-12B-IT | 16 | 16/48 | 0.07 | 2421.8088671875 | 34597.27 |

## 3. Per-vector results

Vectors are paper construction seeds 1–5. Decoding seed is pinned at 12345 for every run,
baselines included.

### Llama-3.1-8B-Instruct, layer 8, r = 0.15

| Set | v1 | v2 | v3 | v4 | v5 | mean | SD |
|:--|---:|---:|---:|---:|---:|---:|---:|
| Medium validation | 30.96 | 34.52 | 24.75 | 34.01 | 34.02 | 31.65 | 4.11 |
| High stakes | 28.22 | 26.73 | 26.89 | 27.40 | 26.57 | 27.16 | 0.67 |
| Astronomical | 25.41 | 26.07 | 24.29 | 25.57 | 24.48 | 25.17 | 0.76 |
| Steals | 65.35 | 64.81 | 67.82 | 64.63 | 64.28 | 65.38 | 1.42 |

### Gemma-3-12B-IT, layer 16, r = 0.07

| Set | v1 | v2 | v3 | v4 | v5 | mean | SD |
|:--|---:|---:|---:|---:|---:|---:|---:|
| Medium validation | 62.31 | 63.45 | 62.81 | 61.73 | 62.31 | 62.52 | 0.64 |
| High stakes | 57.20 | 59.45 | 57.04 | 55.72 | 57.00 | 57.28 | 1.35 |
| Astronomical | 42.65 | 41.78 | 39.53 | 43.00 | 44.51 | 42.29 | 1.83 |
| Steals | 57.70 | 57.20 | 56.46 | 53.79 | 58.96 | 56.82 | 1.92 |

Raw counts per run are in `data/per_run_summary.csv`; every individual choice is in
`data/per_situation_choices.csv` (38,400 rows).

---

## 4. Validation gate

The protocol requires the five fresh vectors to pool at least 950 of 1,000 parsed answers on
medium validation before any held-out set may be run.

| Model | Pooled parsed | Gate | Outcome |
|:--|---:|---:|:--|
| Llama-3.1-8B | 983 / 1000 | ≥ 950 | **pass** |
| Gemma-3-12B | 990 / 1000 | ≥ 950 | **pass** |

Both admitted. No per-seed veto was applied and no seed was replaced.

## 5. The Steals labels

Needed to describe the Steals cost correctly. Restricted to two-option situations, where the
choice is exactly Cooperate versus Steal and the label is unambiguous:

| | Llama run | Gemma run |
|:--|---:|---:|
| Parsed two-option situations | 689 | 687 |
| Cooperate is risk-aversion-optimal (CARA) | 100.0% | 100.0% |
| Cooperate is expected-value-optimal | 82.9% | 82.7% |
| Steal chosen | 245 | 295 |
| …of those, risk-aversion-optimal | 0 | 0 |
| …of those, expected-value-optimal | 88 | 56 |

Option counts across the Steals set: 698 two-option, 172 three-option, 58 four-option,
72 five-option. Every situation's alternatives are Cooperate plus Steal only.

**Steal is a dominated option.** It is never risk-aversion-optimal. So a fall in cooperation
on Steals means the model picks a dominated option more often — a real cost, but *less*
caution, not excessive caution. Any description of the Steals column should be checked
against these labels: it is easy to state the direction backwards.

---

## 6. Cost against benefit

| Model | Mean Δ over three Rebels sets | Δ on Steals | Steals cost per point of benefit |
|:--|---:|---:|---:|
| Qwen3-8B | +56.20 | −30.85 | 0.55 |
| Qwen3-14B | +57.55 | −27.60 | 0.48 |
| Qwen3-1.7B | +43.42 | −22.31 | 0.51 |
| Gemma-3-12B | +42.07 | −22.43 | 0.53 |
| Llama-3.1-8B | +16.08 | −5.35 | 0.33 |

The ratio is strikingly stable at roughly 0.5 across four of the five models. Llama is the
outlier in both size and ratio. This is a descriptive observation from five points, not a
fitted relationship — it should not be presented as a law.

---

## 7. Out-of-distribution shape

Effect size by set, as a fraction of the same model's medium-validation effect:

| Model | Medium | High stakes | Astronomical |
|:--|---:|---:|---:|
| Qwen3-8B | 1.00 | 0.98 | 1.59 |
| Qwen3-14B | 1.00 | 0.91 | 1.37 |
| Qwen3-1.7B | 1.00 | 0.96 | 1.00 |
| Gemma-3-12B | 1.00 | 1.02 | 0.77 |
| Llama-3.1-8B | 1.00 | 0.92 | 1.14 |

The Qwen models gain on the most extreme set; Gemma loses about a quarter of its effect
there; Llama gains modestly. The shape of out-of-distribution generalisation is
model-specific and does not track the size of the in-distribution effect. This comparison
could not be made at all while two of the five models were reported as null.

---

## 8. Response length

Steering roughly doubles Gemma's response length and does so without running into the token
budget.

| Model | Set | Baseline median chars | Steered median chars |
|:--|:--|---:|---:|
| Gemma-3-12B | High stakes | 2,434 | 4,284 |
| Llama-3.1-8B | High stakes | — | 970 |

Gemma steered responses: all 1,000 terminated normally, none hit the cap, range
2,858–6,142 characters. Llama steered: 999 of 1,000 terminated normally, range 335–2,733.

Opener diversity, high stakes: Gemma baseline had 42 distinct 60-character openers with the
most common covering 684 of 1,000; steered had 26 distinct with the most common covering
586 of 1,000. The templated phrasing is a property of Gemma, not something the vector
introduces — steering makes it slightly *less* templated.
