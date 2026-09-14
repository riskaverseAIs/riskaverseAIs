# Paper numbers — activation steering, all five models

**Generated 14 September 2026.** Every figure here was recomputed from the per-situation
records in the run JSONs, not copied from a summary file.

Headline metric throughout: **cooperate rate conditional on a parsed answer**. ± is the
sample standard deviation across the five vectors, never a confidence interval.

---

## 1. The corrected table

Rows marked **NEW** replace the published Llama and Gemma rows. Qwen rows are unchanged and
are reproduced here only so the whole table sits in one place.

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
| **Llama-3.1-8B NEW** | Medium validation | 15.90% | **31.65% ± 4.11** | **+15.75** | 98.30% | 31.10% | 0.10% |
| **Llama-3.1-8B NEW** | High stakes | 12.63% | **27.16% ± 0.67** | **+14.54** | 98.06% | 26.64% | 0.26% |
| **Llama-3.1-8B NEW** | Astronomical | 7.21% | **25.17% ± 0.76** | **+17.96** | 95.76% | 24.10% | 1.76% |
| **Llama-3.1-8B NEW** | Steals | 70.73% | **65.38% ± 1.42** | **−5.35** | 98.26% | 64.24% | 0.10% |
| **Gemma-3-12B NEW** | Medium validation | 17.35% | **62.52% ± 0.64** | **+45.18** | 99.00% | 61.90% | 0.10% |
| **Gemma-3-12B NEW** | High stakes | 11.05% | **57.28% ± 1.35** | **+46.23** | 98.70% | 56.54% | 0.32% |
| **Gemma-3-12B NEW** | Astronomical | 7.51% | **42.29% ± 1.83** | **+34.79** | 97.70% | 41.32% | 0.58% |
| **Gemma-3-12B NEW** | Steals | 79.25% | **56.82% ± 1.92** | **−22.43** | 98.42% | 55.92% | 0.76% |

### What the old Llama and Gemma rows said

| Model | Set | Baseline | Steering | Δ |
|:--|:--|---:|---:|---:|
| Llama-3.1-8B (superseded) | Medium validation | 15.90% | 18.82% ± 1.01 | +2.92 |
| Llama-3.1-8B (superseded) | High stakes | 12.53% | 12.20% ± 0.37 | −0.32 |
| Llama-3.1-8B (superseded) | Astronomical | 7.23% | 8.49% ± 0.77 | +1.25 |
| Llama-3.1-8B (superseded) | Steals | 70.89% | 70.03% ± 0.80 | −0.86 |
| Gemma-3-12B (superseded) | Medium validation | 17.35% | 18.10% ± 1.58 | +0.75 |
| Gemma-3-12B (superseded) | High stakes | 11.05% | 11.29% ± 0.38 | +0.23 |
| Gemma-3-12B (superseded) | Astronomical | 7.51% | 6.26% ± 0.52 | −1.24 |
| Gemma-3-12B (superseded) | Steals | 79.25% | 78.67% ± 0.44 | −0.58 |

---

## 2. Configurations

| Model | Layer (0-based) | Depth | Strength r | Alpha | Mean residual norm at layer |
|:--|---:|:--|---:|---:|---:|
| Llama-3.1-8B-Instruct | **8** | 8/32 | **0.15** | **2.446740245819092** | 16.3116 |
| Gemma-3-12B-IT | **16** | 16/48 | **0.07** | **2421.8088671875** | 34597.27 |

### The superseded configurations, and why they found nothing

| Model | Published layer | Published strength | As `r` | Promoted `r` | Factor below |
|:--|---:|---:|---:|---:|---:|
| Llama-3.1-8B | 12 | r = 0.074325444688 (already a ratio) | 0.0743 | 0.15 | ~2× |
| Gemma-3-12B | 8 | alpha = 16 (raw) | **0.0027** | 0.07 | **~26×** |

The two published configurations were not parameterised the same way. Llama's September
search already worked in ratio units, and its locked setting sat about a factor of two
below the usable window — a near miss, which is why it found a small positive effect
(+2.92 pp on medium) rather than nothing at all. Gemma's search worked in raw alpha over
the integer domain 0–128; its locked alpha of 16 at layer 8, where the mean residual norm
is 5,867, is r = 0.0027, roughly twenty-six times below the promoted setting and about
fifty times below the smallest point on the corrected grid. Gemma's null is a gross
mis-scaling, not a near miss.

Two cross-checks support this reading. The corrected Llama grid at layer 12 gives 16.9% at
r = 0.07 and 21.6% at r = 0.10, bracketing the published 18.82% at r = 0.0743. And Gemma's
largest September alpha, 48, is r = 0.0014 at layer 16 — inside the region where, on the
corrected grid, nothing happens at all.

**One ambiguity.** The three-model protocol records the locked settings as the pairs
`Qwen3-1.7B 5/22; Qwen3-14B 12/48; Gemma3-12B-it 8/16`. Read as layer/strength — consistent
with Qwen3-8B's layer 12 / strength 32 — Gemma is layer 8, strength 16. The reverse reading
(layer 16, strength 8) would give r = 0.00023, an even smaller number that tells the same
story, so the conclusion is unchanged either way. Worth confirming against the Gemma search
records before the sentence goes in the paper.


**`r` is the quantity that matters.** Directions are unit-normalised, so the raw alpha that
gets added to the residual stream is not comparable across models or layers — Gemma's
residual norms are roughly 2,000× Llama's. Define

> `r = alpha / mean_residual_norm_at_layer`

so `r` is the size of the intervention as a fraction of the residual stream it is added to.
The published sweeps searched in raw alpha and therefore placed their grids in different
effective regions for different models. That is the whole explanation for the published
Llama and Gemma nulls; see `writeups/FINDINGS_residual_norms.md`.

Mean residual norm is **prompt-dependent** — it varies by about 17% on Gemma between two
reasonable prompts — so any `r` must be quoted together with the prompt used to measure the
norm. Both values above were measured with the empty system prompt actually used in these
runs.

---

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

---

## 5. Reproduction of the published baselines

Baselines are alpha = 0 through the same hook, run fresh in new processes on a different
instance five days after the published runs.

| Model | Set | Published | New | Gap |
|:--|:--|---:|---:|---:|
| Llama | Medium validation | 15.90% | 15.90% | 0.00 |
| Llama | High stakes | 12.53% | 12.63% | 0.10 |
| Llama | Astronomical | 7.23% | 7.21% | 0.02 |
| Llama | Steals | 70.89% | 70.73% | 0.16 |
| Gemma | Medium validation | 17.35% | 17.35% | 0.00 |
| Gemma | High stakes | 11.05% | 11.05% | 0.00 |
| Gemma | Astronomical | 7.51% | 7.51% | 0.00 |
| Gemma | Steals | 79.25% | 79.25% | 0.00 |

Gemma's four are identical to two decimal places. Llama's four agree to within 0.16 pp.
Both are consistent with the protocols' own statement that exact sampler outcomes are not
guaranteed identical across batching and runtimes; do not describe Llama's as exact.

This is the evidence that the harness has not drifted, and therefore that the change in the
steered numbers is caused by the configuration rather than by the pipeline.

---

## 6. The Steals labels

Needed to describe the Steals cost correctly. Restricted to two-option situations, where the
choice is exactly Cooperate versus Steal and the label is unambiguous:

| | Llama run | Gemma run |
|:--|---:|---:|
| Parsed two-option situations | 689 | 687 |
| Cooperate is risk-aversion-optimal (CARA) | **100.0%** | **100.0%** |
| Cooperate is expected-value-optimal | 82.9% | 82.7% |
| Steal chosen | 245 | 295 |
| …of those, risk-aversion-optimal | **0** | **0** |
| …of those, expected-value-optimal | 88 | 56 |

Option counts across the Steals set: 698 two-option, 172 three-option, 58 four-option,
72 five-option. Every situation's alternatives are Cooperate plus Steal only.

**Steal is a dominated option.** It is never risk-aversion-optimal. So a fall in cooperation
on Steals means the model picks a dominated option more often — a real cost, but *less*
caution, not excessive caution. Any description of the Steals column should be checked
against these labels: it is easy to state the direction backwards.

---

## 7. Cost against benefit

| Model | Mean Δ over three Rebels sets | Δ on Steals | Steals cost per point of benefit |
|:--|---:|---:|---:|
| Qwen3-8B | +56.20 | −30.85 | 0.55 |
| Qwen3-14B | +57.55 | −27.60 | 0.48 |
| Qwen3-1.7B | +43.42 | −22.31 | 0.51 |
| **Gemma-3-12B** | **+42.07** | **−22.43** | **0.53** |
| **Llama-3.1-8B** | **+16.08** | **−5.35** | **0.33** |

The ratio is strikingly stable at roughly 0.5 across four of the five models. Llama is the
outlier in both size and ratio. This is a descriptive observation from five points, not a
fitted relationship — it should not be presented as a law.

---

## 8. Out-of-distribution shape

Effect size by set, as a fraction of the same model's medium-validation effect:

| Model | Medium | High stakes | Astronomical |
|:--|---:|---:|---:|
| Qwen3-8B | 1.00 | 0.98 | 1.59 |
| Qwen3-14B | 1.00 | 0.91 | 1.37 |
| Qwen3-1.7B | 1.00 | 0.96 | 1.00 |
| Gemma-3-12B | 1.00 | 1.02 | **0.77** |
| Llama-3.1-8B | 1.00 | 0.92 | **1.14** |

The Qwen models gain on the most extreme set; Gemma loses about a quarter of its effect
there; Llama gains modestly. The shape of out-of-distribution generalisation is
model-specific and does not track the size of the in-distribution effect. This comparison
could not be made at all while two of the five models were reported as null.

---

## 9. Response length

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
