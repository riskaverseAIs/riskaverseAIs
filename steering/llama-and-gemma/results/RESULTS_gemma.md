# Gemma-3-12B-it — steering sweep, complete result

**14 September 2026.** Medium-stakes validation, first 200 situations. Three construction
seeds (12345 / 23456 / 34567) pooled per point, decoding seed fixed at 12345. Steering
applied at all positions with unit-normalised directions reused from the September build,
so the vectors are bit-identical to the ones the original runs used.

Strength is `r = alpha / mean_residual_norm_at_layer`. Gemma's residual norms vary by a
factor of thirteen across the stack (5,867 at layer 8, 75,780 at layer 24), which is why
a single alpha cannot be compared across layers and why the September sweep went wrong.
Norms measured by `measure_norms.py`; see the addendum in FINDINGS_residual_norms.md on
why the measuring prompt must be stated.

Promotion rule, unchanged: **maximise cooperation among parsed answers subject to a 95%
pooled parse floor.** Nothing else gates eligibility.

## Headline

The published Gemma null is not a weak effect. It is a mis-scaled one.

The control reproduces the published baseline exactly: **17.35%**. The promoted setting
reaches **62.67% cooperation at 98.67% parse** — a **45.3 point** move, about 29 standard
errors. September's largest alpha was 48, which at layer 16 is r = 0.0014: roughly fifty
times below the *smallest* point on this grid. The whole sweep was run inside the region
where nothing happens.

## Promoted candidate

**Layer 16, r = 0.07, alpha = 2421.8089 — 62.67% cooperation, 98.67% parse.**

Unlike Llama, this one is not on a knife edge. All four refinement neighbours are
*eligible* and all score lower:

| probe | cooperate | parse | eligible |
|:--|---:|---:|:--:|
| r = 0.0595 (weaker) | 58.5% | 98.5% | yes |
| **r = 0.07** | **62.7%** | **98.7%** | **yes** |
| r = 0.0826 (stronger) | 61.0% | 97.0% | yes |
| layer 15 | 44.3% | 97.8% | yes |
| layer 17 | 41.4% | 98.3% | yes |

This is what a real optimum looks like: the neighbourhood is valid and lower. The Llama
leader had the opposite signature — every neighbour failed the floor — which is why it was
set aside there and why it does not need to be set aside here.

Recomputed independently from the three raw per-seed files rather than the summary:
198 / 195 / 199 parsed out of 200 each (592/600 = 98.67%), and 126 / 124 / 121 cooperate
(371/592 = 62.67%). Both figures match the sweep's own table exactly, and the per-seed
cooperation rates — 63.6%, 63.6%, 60.8% — are tight enough that no single construction
seed is carrying the result.

## Layer scan

Cooperation, with parse rate in brackets. ✗ marks points below the floor.

| layer | r = 0.045 | r = 0.07 | r = 0.10 |
|---:|---:|---:|---:|
| 8  | 28.3% (97.2%) | 42.8% (96.5%) | 63.9% (92.8% ✗) |
| 12 | 32.2% (97.8%) | 57.5% (96.2%) | 56.1% (96.5%) |
| **16** | 59.7% (98.5%) | **62.7% (98.7%)** | 50.4% (21.5% ✗) |
| 20 | 23.4% (99.8%) | 25.4% (99.2%) | 31.6% (96.7%) |
| 24 | 22.0% (99.2%) | 23.7% (98.3%) | 27.6% (96.5%) |

Layer 16 is an interior peak, not the end of a trend. Layers 20 and 24 are close to inert:
they sit in the low 20s against a 17.35% baseline while parsing at 96–99%, so nothing is
straining — the direction simply does little there. Layers 8 and 12 respond, but need more
strength to do it and start losing parse before they arrive. Layer 8 at r = 0.10 actually
reaches 63.9%, nominally the highest number in the sweep, but at 92.8% parse it is
ineligible.

The peak is sharp. One layer either side of 16 costs roughly eighteen points — 44.3% at
layer 15, 62.7% at 16, 41.4% at 17 — with parse healthy throughout. On ~600 pooled
responses the standard error is about 2 points, so an 18-point drop is around nine standard
errors and is not noise. It is still worth flagging for the audit: directions are built
per-layer, so this may reflect the quality of the layer-16 direction rather than a property
of the residual stream at that depth.

## Strength scan, layer 16

| r | alpha | cooperate | parse | token-limit | looping |
|---:|---:|---:|---:|---:|---:|
| 0     | 0      | 17.3% | 98.0% | 1.5% | 0.5% |
| 0.02  | 692    | 36.3% | 98.7% | 0.5% | 0.5% |
| 0.03  | 1,038  | 44.9% | 98.0% | 0.3% | 0.3% |
| 0.045 | 1,557  | 59.7% | 98.5% | 0.0% | 0.0% |
| **0.07** | **2,422** | **62.7%** | **98.7%** | 0.3% | 0.3% |
| 0.10  | 3,460  | (50.4%) | 21.5% ✗ | 99.7% | 40.2% |
| 0.15  | 5,190  | (100.0%) | 0.2% ✗ | 100.0% | 97.8% |
| 0.22–1.50 | 7,611–51,896 | — | 0.0% ✗ | 100.0% | 0.2–100% |

The usable window closes hard between r = 0.07 and r = 0.10 — 98.7% parse to 21.5% across
a single grid step.

## Two things the collapse branch shows

**The denominator artifact, at its limit.** Cooperation is computed among parsed answers.
At r = 0.15 the sweep reports **100.0% cooperation on a 0.2% parse rate** — that is one
readable response out of 600, which happened to be a Cooperate. Read off a results table
without its parse column, that is the best number in the entire study; it is in fact a
completely destroyed model and a sample size of one. This is the case the parse floor
exists to exclude, and the reason the floor must be checked before the cooperation number
is read at all.

**Gemma and Llama break differently.** Llama loops into short repetitive output and
terminates. Gemma runs the full 4,096-token budget on nearly every response without
producing an answer — and often without looping at all: at r = 0.22 the token-limit rate is
100% while the looping rate is 8.5%, and at r = 0.63 it is 100% against 0.2%. So "parse
failure" is not one phenomenon across model families, and a protocol that diagnoses
collapse by looking for repetition would miss Gemma's failure entirely. It also has a
practical cost: Gemma's collapsed points took 31–146 minutes each against Llama's 8, because
every response runs to the cap.

## Next

Independent audit of the results table, then five-seed paper-facing runs (seeds 1–5,
medium / high / astronomical / Steals) at layer 16, r = 0.07, alpha = 2421.8089.
