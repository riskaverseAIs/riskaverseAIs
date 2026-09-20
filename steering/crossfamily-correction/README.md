# Cross-family steering correction — Llama-3.1-8B and Gemma-3-12B

Corrected activation-steering results for the two non-Qwen model families, with the full
hyperparameter search, the paper-facing runs, every individual answer, and the code.

## The correction

An earlier sweep reported activation steering as essentially null on Llama-3.1-8B
(+2.92 pp cooperate rate on medium-stakes validation) and Gemma-3-12B (+0.75 pp), against
large effects on three Qwen3 models.

That was an artifact of how strength was parameterised. Steering adds `alpha * direction` to
the residual stream, and the direction is unit-normalised — so `alpha` on its own says
nothing about how large the intervention is relative to the activations it perturbs. Mean
residual-stream norms differ enormously: 1.25 to 57 across Llama's layers, 1,105 to 161,797
across Gemma's, with Gemma roughly 2,000x Llama at comparable depth. A grid over raw `alpha`
therefore lands in a different effective region for every model and every layer.

Re-expressing strength as

    r = alpha / mean_residual_norm_at_layer

and re-running the same search protocol finds large effects in both models. The published
Gemma setting works out to r = 0.0027, about twenty-six times below the promoted value.

| model | set | baseline | superseded | **corrected** |
|---|---|---:|---:|---:|
| Llama-3.1-8B | medium validation | 15.90% | 18.82% (+2.92) | **31.65% +/- 4.11 (+15.75)** |
| Llama-3.1-8B | high stakes | 12.63% | 12.20% (-0.32) | **27.16% +/- 0.67 (+14.54)** |
| Llama-3.1-8B | astronomical | 7.21% | 8.49% (+1.25) | **25.17% +/- 0.76 (+17.96)** |
| Llama-3.1-8B | steals | 70.73% | 70.03% (-0.86) | **65.38% +/- 1.42 (-5.35)** |
| Gemma-3-12B | medium validation | 17.35% | 18.10% (+0.75) | **62.52% +/- 0.64 (+45.18)** |
| Gemma-3-12B | high stakes | 11.05% | 11.29% (+0.23) | **57.28% +/- 1.35 (+46.23)** |
| Gemma-3-12B | astronomical | 7.51% | 6.26% (-1.24) | **42.29% +/- 1.83 (+34.79)** |
| Gemma-3-12B | steals | 79.25% | 78.67% (-0.58) | **56.82% +/- 1.92 (-22.43)** |

Cooperate rate among parsed answers. `+/-` is the sample standard deviation across five
independently constructed direction vectors, not a confidence interval.

## Configurations

| model | layer (0-based) | r | alpha | mean residual norm at layer |
|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 8 | 0.15 | 2.446740245819092 | 16.3116 |
| Gemma-3-12B-IT | 16 | 0.07 | 2421.8088671875 | 34597.27 |

Evaluated with chains of thought enabled, an empty system prompt, and the direction added
at all token positions. The direction vectors are in the companion model archive; this
directory holds results and code only.

Mean residual norm is prompt-dependent — about 17% variation on Gemma between two reasonable
measuring prompts — so a ratio is only meaningful alongside the prompt used to measure the
norm. Both values above use the empty system prompt these runs use. See
`results/FINDINGS_residual_norms.md`.

## Layout

```
RESULTS.md            every figure, with superseded values alongside
METHODS.md            method, protocol, declared deviations, limitations

results/
  per_set_summary.csv/.json   8 rows: model x set, with SDs and diagnostics
  per_run_summary.csv         48 rows: one per run, raw counts
  per_situation_choices.csv   38,400 rows: every answer, with CARA/linear labels
  per_run_breakdowns.csv      evaluator's own subset / format / stakes breakdowns
  *_PAPER_RESULTS.json        aggregate output of the paper runs
  *_CANDIDATES.json           every configuration tried in the search, with scores
  *_RESIDUAL_NORMS.json       mean residual norm per layer
  RESULTS_paper_*.md          full write-ups of the five-vector runs
  RESULTS_llama.md            hyperparameter search write-ups
  RESULTS_gemma.md
  FINDINGS_residual_norms.md  why raw alpha is not comparable; prompt-dependence

code/
  run_paper.py          five-vector runner, including the validation gate
  run_paper_chain.sh    the exact launch script
  run_sweep.py          hyperparameter search; also defines the shared scorer
  measure_norms.py      residual-norm measurement
  hook_audit.py         independent check that the steering hook does what it claims

provenance/
  PROVENANCE.json       versions, pinned revisions, vector hashes, extraction records
  DATASETS.json         the four evaluation CSVs with sha256
  paper_chain.log       complete run log with timings
```

## Protocol

- Search on medium-stakes validation, first 200 situations, three construction seeds pooled
  (600 answers per candidate), decoding seed fixed at 12345.
- Promotion: highest cooperate rate among parsed answers, subject to a **95% pooled parse
  floor**. That floor is the only eligibility gate.
- Paper runs: five independently constructed vectors (construction seeds 1-5), one alpha-0
  baseline through the same hook, on four sets. 24 runs and 19,200 answers per model.
- Admission gate: the five medium-validation runs must pool >= 950/1000 parsed before any
  held-out set starts. Llama pooled 983, Gemma 990.
- Generation: vLLM, thinking enabled, temperature 0.6, top-p 0.95, top-k 20, max 4096 new
  tokens, soft 800-token reasoning instruction, bfloat16, context 8192, eager execution while
  steering, prefix caching disabled, batch 256.

Pinned revisions: `meta-llama/Llama-3.1-8B-Instruct` at `0e9e39f2...`, `google/gemma-3-12b-it`
at `96b6f1ec...`. Full hashes in `provenance/PROVENANCE.json`.

## Reproducing the headline from the raw data

```bash
python3 - <<'EOF'
import csv, collections, statistics
rows = list(csv.DictReader(open("results/per_situation_choices.csv")))
agg = collections.defaultdict(lambda: [0, 0])
for r in rows:
    a = agg[(r["model"], r["dataset"], r["run"])]
    a[0] += int(r["parsed"]); a[1] += int(r["cooperate"])
for m in ["llama", "gemma"]:
    for d in ["medium_stakes_validation", "high_stakes_test",
              "astronomical_stakes_deployment", "steals_test"]:
        b = agg[(m, d, "baseline")]
        rates = [100 * agg[(m, d, "seed_%d" % i)][1] / agg[(m, d, "seed_%d" % i)][0]
                 for i in range(1, 6)]
        print("%-6s %-32s baseline %6.2f  steering %6.2f +- %4.2f" % (
            m, d, 100 * b[1] / b[0], statistics.mean(rates), statistics.stdev(rates)))
EOF
```

## Two things to read before quoting a number

**Report parse rate beside cooperate rate.** As parse rate falls, cooperation among parsed
answers rises spuriously. The worst case observed here is Gemma at r = 0.15: 100.0%
cooperation on a 0.2% parse rate — one response out of 600.

**Steal is a dominated option.** In every parsed two-option Steals situation, Cooperate is
the risk-aversion-optimal choice; Steal is risk-aversion-optimal in none of them. So a fall
in cooperation on the Steals set means the model picks a dominated option more often. That is
a cost, but it is *less* caution, not excessive caution. `RESULTS.md` §6 has the counts.
