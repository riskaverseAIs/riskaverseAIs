# Llama-3.1-8B-Instruct — five-vector paper-facing runs

**14 September 2026.** Five independently constructed steering vectors (paper seeds 1–5),
decoding seed pinned at 12345, batch 256. One baseline at alpha 0 through the same hook
plus five vector runs on each of four sets: medium validation (first 200 situations),
high stakes, astronomical stakes, and Steals (1,000 each). 24 runs, 19,200 answers.

Configuration: **layer 8, r = 0.15, alpha = 2.446740245819092.**

The five vectors already existed from the September all-layer extraction — the run that
selected layer 12 retained every other extracted layer — so these are the same audited
vectors the published table used, read at a different layer. No new construction.

Medium validation ran first and the five vector runs pooled **983/1000 parsed**, clearing
the protocol's 950/1000 gate, which is what admitted the three held-out sets.

## Headline

| set | baseline | published steering | **new steering** | published Δ | **new Δ** |
|:--|---:|---:|---:|---:|---:|
| Medium validation | 15.90% | 18.82% ± 1.01 pp | **31.65% ± 4.11 pp** | +2.92 pp | **+15.75 pp** |
| High stakes | 12.63% | 12.20% ± 0.37 pp | **27.16% ± 0.67 pp** | −0.32 pp | **+14.53 pp** |
| Astronomical stakes | 7.21% | 8.49% ± 0.77 pp | **25.17% ± 0.76 pp** | +1.25 pp | **+17.96 pp** |
| Too-risk-averse / Steals | 70.73% | 70.03% ± 0.80 pp | **65.38% ± 1.42 pp** | −0.86 pp | **−5.35 pp** |

± is the sample standard deviation across the five vectors. Cooperation is conditional on
a parsed answer.

**All four baselines reproduce the published values** — 15.90 vs 15.90, 12.63 vs 12.53,
7.21 vs 7.23, 70.73 vs 70.89. These were run fresh, in new processes, on a different
instance, five days later. That is the strongest available evidence that nothing in this
pipeline has drifted from September, and therefore that the change in the steered numbers
is caused by the configuration and not by the harness.

## The effect transfers

The published result did not merely find a small effect on Llama; on high stakes it found
a slightly negative one, which reads as the vector failing to generalise out of
distribution. The corrected configuration moves high stakes by +14.53 pp and astronomical
stakes by +17.96 pp — the latter being the largest move of the four sets and more than
triple the baseline rate.

So the effect is not confined to the set the configuration was tuned on. If anything it is
strongest on the most out-of-distribution set.

## Per-vector detail

| set | per-vector cooperation | pooled parse | coop / all attempts | token limit |
|:--|:--|---:|---:|---:|
| Medium validation | 30.96, 34.52, 24.75, 34.01, 34.02 | 98.30% | 31.10% | 0.10% |
| High stakes | 28.22, 26.73, 26.89, 27.40, 26.57 | 98.06% | 26.64% | 0.26% |
| Astronomical | 25.41, 26.07, 24.29, 25.57, 24.48 | 95.76% | 24.10% | 1.76% |
| Steals | 65.35, 64.81, 67.82, 64.63, 64.28 | 98.26% | 64.24% | 0.10% |

**The medium standard deviation of 4.11 pp overstates vector-to-vector variability** and
should not be read as instability. Each vector sees only 200 situations there against
1,000 on the held-out sets, so the per-vector standard error is roughly twice as large.
The spread is also driven by one vector (seed 3, 24.75%), which sits about 2.7 standard
errors below the others on medium but is unremarkable on all three held-out sets. On those
sets, with five times the sample, the spread collapses to 0.67–1.42 pp.

## The Steals result is a cost, and the sign matters

Cooperation on Steals falls by 5.35 pp. Reading that correctly requires knowing which
option the dataset marks as correct, so the labels were checked directly:

| set | option | risk-aversion-optimal (CARA) | expected-value-optimal |
|:--|:--|---:|---:|
| High stakes | Cooperate | 75.8% | 0.0% |
| High stakes | Rebel | 5.8% | 93.0% |
| Steals | Cooperate | **91.2%** | **92.1%** |
| Steals | Steal | **0.0%** | 51.7% |

The two Steals rows pool situations with two, three, four, and five options. Restricting to
the 689 parsed two-option situations, where the choice is exactly Cooperate versus Steal and
the label is unambiguous, Cooperate is risk-aversion-optimal in **100.0%** of them and
expected-value-optimal in 82.9%, and Steal was chosen 245 times without once being
risk-aversion-optimal. Use those figures; they are the same in the Gemma run (100.0% and
82.7%, 0 of 295).

On the Rebels sets there is a genuine trade-off — Cooperate is risk-aversion-optimal,
Rebel is expected-value-optimal — so raising cooperation is the intended effect.

**On Steals there is no trade-off.** Cooperate is optimal under both criteria, and Steal is
risk-aversion-optimal in zero of 984 parsed situations. Steal is dominated. A fall in
cooperation there means the model selects a dominated option more often.

This is a real cost of the intervention and should be reported as one. But it is not
excessive caution — it is *less* caution, in the one set where caution is unambiguously
correct. Describing a Steals drop as the intervention inducing broad caution "including
where caution is excessive" has the direction backwards, read against these labels: the
model is becoming less cautious, not more. The underlying finding survives either way and is arguably worse under
the corrected reading: a vector that degrades judgment is a more serious objection than one
that is merely over-tuned.

Llama's cost here is about one sixth of Qwen's (−5.35 pp against −30.85 pp).

## Selection bias in the search number

The search reported **34.5%** at this configuration on medium validation. The five fresh
vectors give **31.65%**. The search value sits at the top of the fresh-vector range, which
is expected: the configuration was selected using those three search seeds, so its estimate
is optimistically biased. The fresh-vector number is the one for the paper. This is exactly
what constructing new vectors is for, and is worth a sentence in the methods.

## Response quality

Checked on the high-stakes seed-1 run (1,000 responses): 999 terminated normally, one hit
the token cap. Median response 970 characters, range 335–2,733. Thirty-six distinct openers
among the 278 cooperates, the most frequent appearing 62 times.

The reasoning is on-topic but not good — the sampled response invents a non-standard
utility function and produces some incorrect intermediate values. That is a property of
Llama-3.1-8B rather than of the steering, but it means the transcripts do not support a
claim that the vector makes the model reason about risk *well*. What they support is that
it changes which option is chosen, in coherent on-topic text.

## Provenance

Results at `artifacts/llama_PAPER_RESULTS.json`; per-run JSON and logs under
`/home/ubuntu/paper-llama/<set>/<baseline|seed_N>/`. Scoring imports `score_result` from
`run_sweep.py`, so these runs are scored by byte-identical code to the search, including
its reconciliation against the evaluator's own `cooperate_rate`.
