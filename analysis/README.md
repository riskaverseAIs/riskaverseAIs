# Analysis

Post-hoc analysis of saved evaluation output. Nothing here needs a GPU: every
script takes the per-situation results file that `evaluation/evaluate.py` writes
under `--save_responses` and works from that.

| file | what it does |
|---|---|
| `analyze_benchmark.py` | scores each surface heuristic against the benchmark's own labels — what a rule could in principle achieve |
| `analyze_model_vs_heuristics.py` | item-level agreement between a model's choices and each heuristic — which rule a model is actually running |
| `generate_magnitude_sweep.py` | builds the matched-family prize-magnitude sweep from the astronomical-stakes set |
| `magnitude_sweep_figure.tex` | the `pgfplots` source for the sweep figure in the paper |

## Are the models running a crude heuristic?

The worry is that a model reaching a high Cooperate rate is not risk-averse at
all, but following a surface rule such as "pick the option with the smallest
numbers" or "the option with the huge number is a trap". Scoring a rule against
the benchmark labels cannot settle this, because two rules that score the same
in aggregate can disagree on half the individual situations. So the test is
item-level agreement:

```bash
python analyze_model_vs_heuristics.py \
  --results   /path/to/results_sft_seed1.json \
  --benchmark ../evaluation/data/2026_03_22_astronomical_stakes_deployment_set_1000_Rebels.csv
```

A model whose agreement with one rule is near 100% is running that rule
whatever its chain of thought says. A model whose agreement with every rule
sits near the chance rate implied by the option counts is doing something none
of the rules capture, which is what the paper's claims require.

## The prize-magnitude sweep

Within a family everything is held fixed — the probability vector, the initial
wealth, the whole Cooperate option, the prompt template, the option order — and
only the displayed takeover payoff changes. So the Cooperate rate as a function
of magnitude separates three behaviours: it rises with magnitude under the
"huge number is a trap" heuristic, stays flat under the bounded CARA target, and
falls under expected-value reasoning.

```bash
python generate_magnitude_sweep.py \
  --source ../evaluation/data/2026_03_22_astronomical_stakes_deployment_set_1000_Rebels.csv \
  --out    magnitude_sweep.csv
```

The output keeps the source schema plus `family_id` and `takeover_magnitude`,
so it can be passed straight to `evaluation/evaluate.py`.
