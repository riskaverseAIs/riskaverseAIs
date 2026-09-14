# Gemma-3-12B-IT — five-vector paper-facing runs

**14 September 2026.** Five independently constructed steering vectors (paper seeds 1–5),
decoding seed pinned at 12345, batch 256. One baseline at alpha 0 through the same hook
plus five vector runs on each of four sets: medium validation (first 200 situations),
high stakes, astronomical stakes, and Steals (1,000 each). 24 runs, 19,200 answers.

Configuration: **layer 16, r = 0.07, alpha = 2421.8088671875.**

As with Llama, the five vectors already existed from the September all-layer extraction,
so these are the same audited vectors the published table used, read at a different layer.
No new construction.

Medium validation ran first and the five vector runs pooled **990/1000 parsed**, clearing
the protocol's 950/1000 gate, which is what admitted the three held-out sets.

## Headline

| set | baseline | published steering | **new steering** | published Δ | **new Δ** |
|:--|---:|---:|---:|---:|---:|
| Medium validation | 17.35% | 18.10% ± 1.58 pp | **62.52% ± 0.64 pp** | +0.75 pp | **+45.17 pp** |
| High stakes | 11.05% | 11.29% ± 0.38 pp | **57.28% ± 1.35 pp** | +0.23 pp | **+46.23 pp** |
| Astronomical stakes | 7.51% | 6.26% ± 0.52 pp | **42.29% ± 1.83 pp** | −1.24 pp | **+34.78 pp** |
| Too-risk-averse / Steals | 79.25% | 78.67% ± 0.44 pp | **56.82% ± 1.92 pp** | −0.58 pp | **−22.43 pp** |

± is the sample standard deviation across the five vectors. Cooperation is conditional on
a parsed answer.

**All four baselines reproduce the published values exactly** — 17.35, 11.05, 7.51, 79.25,
against published 17.35, 11.05, 7.51, 79.25 — identical to two decimal places on all four.

Llama's four baselines are close but not identical (15.90 vs 15.90, 12.63 vs 12.53, 7.21 vs
7.23, 70.73 vs 70.89; largest gap 0.16 pp). That is the expected level of agreement, not a
problem: the protocols state that exact sampler outcomes are not guaranteed identical across
batching and runtimes. Gemma's landing exactly is the surprise, not Llama's near-miss.

Either way the point holds. Eight baselines run in new processes on a different instance five
days later land on or within 0.16 pp of September's, so the harness has not drifted and the
change in the steered numbers is caused by the configuration alone.

## The published Gemma null was an artifact of where the grid was placed

The published sweep searched in raw alpha, which is not comparable across models or layers
because the directions are unit-normalised and the residual stream is not. Expressed as a
fraction of the mean residual norm at the layer, the published settings sat far from the
region where this model responds. At layer 16 and r = 0.07 the model moves 45 points on the
set it was tuned on and 46 on held-out high stakes.

The effect is larger than Llama's by a factor of about three, and comparable to Qwen3-8B's
(+47.28 medium, +46.13 high). Gemma is not a model that resists steering.

## The effect attenuates on the most extreme set — and Llama's does not

| set | Gemma Δ | Llama Δ |
|:--|---:|---:|
| Medium validation | +45.17 pp | +15.75 pp |
| High stakes | +46.23 pp | +14.53 pp |
| Astronomical stakes | **+34.78 pp** | **+17.96 pp** |

Gemma holds its effect from medium to high stakes and then loses about a quarter of it on
astronomical stakes. Llama, by contrast, shows its *largest* move there. Qwen3-8B went the
other way again, gaining on astronomical stakes (+75.18 against +46.13 on high stakes).

So the shape of out-of-distribution generalisation is model-specific and does not follow
the size of the in-distribution effect. That is worth a line in the paper: it is a point
the three-model comparison can now make and could not make when two of the three were null.

## Per-vector detail

| set | per-vector cooperation | pooled parse | coop / all attempts | token limit |
|:--|:--|---:|---:|---:|
| Medium validation | 62.31, 63.45, 62.81, 61.73, 62.31 | 99.00% | 61.90% | 0.10% |
| High stakes | 57.20, 59.45, 57.04, 55.72, 57.00 | 98.70% | 56.54% | 0.32% |
| Astronomical | 42.65, 41.78, 39.53, 43.00, 44.51 | 97.70% | 41.32% | 0.58% |
| Steals | 57.70, 57.20, 56.46, 53.79, 58.96 | 98.42% | 55.92% | 0.76% |

Vector-to-vector agreement is much tighter than Llama's, including on medium validation
where the 200-situation sample inflated Llama's spread to 4.11 pp. Gemma's medium spread is
0.64 pp on the same sample size. This is consistent with what the search found: Gemma's
optimum at layer 16 is a genuine interior peak with all four refinement neighbours eligible
and lower, whereas Llama's was a knife-edge where neighbouring settings failed the parse
floor outright.

Parse rates stay high everywhere and the token-limit rate never exceeds 0.76%, so none of
these numbers are inflated by the denominator effect that made the far end of the sweep
look artificially good.

## The Steals result is a cost, and the sign matters

Cooperation on Steals falls by 22.43 pp. Reading that correctly requires knowing which
option the dataset marks as correct, so the labels were checked directly in the run records.

Restricting to the two-option Steals situations, where the choice is exactly Cooperate
versus Steal and the label is unambiguous:

| | Gemma run | Llama run |
|:--|---:|---:|
| parsed two-option situations | 687 | 689 |
| Cooperate is risk-aversion-optimal (CARA) | **100.0%** | **100.0%** |
| Cooperate is expected-value-optimal | 82.7% | 82.9% |
| Steal chosen, and risk-aversion-optimal | **0 of 295** | **0 of 245** |

**On Steals there is no trade-off.** Cooperate is the risk-aversion-optimal choice in every
two-option situation and the expected-value-optimal choice in most of them; Steal is
risk-aversion-optimal in none. Steal is dominated. A fall in cooperation there means the
model selects a dominated option more often.

This is a real cost of the intervention and should be reported as one. But it is not
excessive caution — it is *less* caution, in the one set where caution is unambiguously
correct. Describing a Steals drop as the intervention inducing broad caution "including
where caution is excessive" has the direction backwards, read against these labels: the
model is becoming less cautious, not more. The underlying finding survives either way and is arguably worse under
the corrected reading: a vector that degrades judgment is a more serious objection than one
that is merely over-tuned.

(These per-situation figures supersede the coarser 91.2% / 92.1% quoted in the Llama
write-up, which pooled situations with three, four, and five options together with the
two-option ones. The claim that matters is unchanged and is exact in both: Steal is never
the risk-aversion-optimal choice.)

Ranked by cost per unit of benefit, Gemma sits between the other two: it pays 22.43 points
on Steals for roughly 42 points of average benefit on the three Rebels sets, where Qwen3-8B
pays 30.85 for about 56 and Llama pays 5.35 for about 16.

## Selection bias in the search number

The search reported **62.67%** at this configuration on medium validation. The five fresh
vectors give **62.52%** — an optimism of 0.15 pp, effectively nil.

Llama's search number was optimistic by 2.85 pp on the same comparison. The difference is
explained by the shape of the two optima: Llama's configuration was selected at a narrow
peak where the search seeds could be lucky, Gemma's at a broad one where they could not.
So the bias is a property of the search landscape, not a general property of the method,
and the fresh-vector numbers are the ones for the paper in both cases.

## Response quality

Checked on the high-stakes seed-1 run (1,000 responses): **all 1,000 terminated normally**,
none hit the token cap. Median response 4,284 characters, range 2,858–6,142.

Steering roughly doubles response length — the baseline's median is 2,434 characters. The
model reasons at more length about the gamble before answering, and does so without running
into the budget.

Openers are heavily stereotyped, but *less* so under steering than at baseline: 26 distinct
60-character openers among the steered responses with the most common covering 586 of 1,000,
against 42 distinct and 684 of 1,000 at baseline. The template is a property of Gemma, not
something the vector introduces.

## Provenance

Results at `artifacts/gemma_PAPER_RESULTS.json`; per-run JSON and logs under
`/home/ubuntu/paper-gemma/<set>/<baseline|seed_N>/`. Scoring imports `score_result` from
`run_sweep.py`, so these runs are scored by byte-identical code to the search. Every figure
in this document was recomputed independently from the per-situation records in the run
files rather than read from the summary, and matched.
