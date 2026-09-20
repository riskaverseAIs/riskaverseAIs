# Llama-3.1-8B-Instruct — steering sweep, complete result

**13 September 2026.** Medium-stakes validation, first 200 situations. Three
construction seeds (12345 / 23456 / 34567) pooled per point, decoding seed fixed at
12345. Steering applied at all positions with unit-normalised directions reused from
the September build, so the vectors are bit-identical to the ones the original runs used.

Strength is `r = alpha / mean_residual_norm_at_layer` — the fraction of the residual
stream the added direction represents. Norms measured by `measure_norms.py`; see the
addendum in FINDINGS_residual_norms.md on why the measuring prompt must be stated.

Promotion rule, unchanged from the paper's other sweeps: **maximise cooperation among
parsed answers subject to a 95% pooled parse floor.** Nothing else gates eligibility.

## Headline

Llama responds to steering. The published null came from never sampling the range where
it works.

September tested alpha 0.5 and alpha 16. Both reproduce exactly here: alpha 0.5 is
r = 0.03, which moves cooperation 2.6 points; alpha 16 is just above r = 0.85, which
gives 0% parse and 100% looping. The entire usable window lies between them, at
**alpha 1.8 to 2.7**, and was never sampled.

## Strength scan, layer 12

| r | alpha | cooperate | parse | token-limit | looping |
|---:|---:|---:|---:|---:|---:|
| 0     | 0     | 15.9% | 97.5% | 0.0% | 0.0% |
| 0.02  | 0.36  | 17.8% | 96.5% | 0.0% | 0.0% |
| 0.03  | 0.54  | 18.5% | 96.5% | 0.0% | 0.0% |
| 0.045 | 0.81  | 17.8% | 95.5% | 0.0% | 0.0% |
| 0.07  | 1.27  | 16.9% | 94.8% ✗ | 0.0% | 0.0% |
| 0.10  | 1.81  | 21.6% | 95.5% | 0.3% | 0.0% |
| 0.15  | 2.71  | 28.6% | 96.0% | 1.2% | 1.0% |
| 0.22  | 3.98  | (34.8%) | 55.0% ✗ | 44.5% | 23.8% |
| 0.32  | 5.78  | (47.6%) | 21.0% ✗ | 94.0% | 83.5% |
| 0.45–1.50 | 8.1–27.1 | — | 0.0% ✗ | 100.0% | ~100% |

## Layer scan at r = 0.15

| layer | cooperate | parse | note |
|---:|---:|---:|:--|
| 6  | (39.0%) | 83.3% ✗ | 16.3% refusals — see below |
| 7  | 28.8% | 97.2% | post-hoc |
| **8**  | **34.5%** | **99.0%** | peak; post-hoc neighbours on both sides |
| 10 | 31.0% | 97.2% | |
| 12 | 28.6% | 96.0% | preregistered centre |
| 14 | 19.9% | 93.2% ✗ | |
| 16 | 23.2% | 96.2% | |

Layers 6 and 7 were added after seeing the gradient and are flagged `post_hoc` in
CANDIDATES.json. They were run to find where the shallow trend turns over. It turns
over at layer 8, which has clean neighbours on both sides and the joint-best parse
rate in the sweep.

## The candidate for the paper

**Layer 8, r = 0.15, alpha = 2.4467, 34.5% cooperation, 99.0% parse.**

The promotion rule, applied mechanically, returns a different point: layer 10 at
r = 0.22, which scores 38.1% cooperation at a parse rate of exactly 95.0%. That point
is not being used, and the reason is worth stating in the paper.

| | cooperate | parse | margin over floor | regime |
|:--|---:|---:|---:|:--|
| layer 10, r = 0.22 (rule's leader) | 38.1% | 95.0% | 0 responses | unstable |
| **layer 8, r = 0.15** (chosen) | **34.5%** | **99.0%** | 30 responses | stable |

Layer 10 at r = 0.22 passes on exactly 570 parsed responses out of 600. One fewer would
have disqualified it. Refinement probed all four directions around it — weaker,
stronger, shallower, deeper — and every neighbour failed the floor, including a *weaker*
setting at the same layer (r = 0.187, 94.7%), which is a two-response difference in the
wrong direction. At r = 0.22 the parse rate across seven layers reads
87.3 / 81.7 / **95.0** / 87.0 / 55.0 / 77.8 / 91.7: layer 10 is the only one above the
floor, in a band scattering over forty points with no orderly relation to depth.

A point whose entire neighbourhood is ineligible is the top of the noise, not a peak.
Layer 8 at r = 0.15 sits where every measurement behaves smoothly, is the maximum in
layer space, and holds the joint-best parse rate in the sweep.

Both are large moves off a 15.9% baseline and both refute the published null, so the
paper's claim never depended on the choice — only which number goes in the table, and
how likely that number is to survive re-running on paper seeds 1–5.

## Refusals

Refusal rate is 0.0% at every candidate except layer 6 (16.3%) and a trace at
layer 8 / r = 0.22 (0.2%). So every other parse failure in this sweep is generation
breaking down — running to the token cap, looping — and not the model declining to answer.

Layer 6 is different in kind. Its unparsed responses all terminate normally and are
refusals ("I cannot provide financial advice", "I cannot provide a personal choice in a
hypothetical scenario"). Its 39.0% is therefore computed over a non-random 83.3% subset
and is not the same quantity as the other rows. Worth a sentence in the paper: at shallow
layers the steering direction begins engaging refusal behaviour rather than risk
preference, which suggests the direction is not cleanly encoding the intended concept
there.

## Reading the collapsed points

Cooperation is computed among parsed answers. As parse falls the denominator stops being
a random sample, so the cooperation rate *rises* through the collapse — 28.6%, 34.8%,
47.6% — while the model is getting worse. This is the artifact the parse floor exists to
exclude, and it is why the floor must be checked before the cooperation number is read.

## Next

Five-seed paper-facing runs (seeds 1–5, medium / high / astronomical / Steals) at
layer 8, r = 0.15, alpha = 2.4467.
