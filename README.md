# Risk-Averse AIs

Code, data, and locked configurations for reproducing the experiments in
**Out-of-Distribution Generalization of Risk Aversion in Language Models**, by
Kristina Zhang, Junior Chinomso Okoroafor, Benjamin Maltbie, Andrew Lin,
Abhitej Bokka, and Elliott Thornley (Massachusetts Institute of Technology).

The trained adapters and steering directions are not in this repository; they
live in the companion model archive, described under
[Model artifacts](#model-artifacts) below.

The repository is organized around one shared evaluation package plus one
directory per intervention:

- `evaluation/`: main generative evaluator, reward-model evaluator, transfer-benchmark helpers, and shared datasets.
- `sft-training/`: supervised fine-tuning code and training data.
- `tie-training/`: supervised tie-training instructions using the shared SFT trainer.
- `dpo-training/`: DPO trainer plus a helper to build the 600-pair lin-only JSONL.
- `reward-model/`: reward-model fine-tuning pipeline and RewardBench 2 evaluation.
- `steering/`: activation-steering instructions, direction-building helpers, the locked
  per-model configurations, and the hyperparameter search.
- `dataset-generation/`: scripts used to generate the paper's benchmark CSVs.
- `analysis/`: post-hoc analysis of saved evaluation output — heuristic-agreement
  checks and the prize-magnitude sweep. No GPU needed.

Each method directory has its own README with the paper-facing commands. The
shared datasets used across methods live in `evaluation/data/`.

## Quick Start

1. Create a Python environment.
2. Install the requirements for the method you want to run.
3. Use the shared datasets in `evaluation/data/`.
4. Evaluate trained adapters with `evaluation/evaluate.py`.

### Reference Environment

The paper's runs used this pinned environment (Python 3.10+, CUDA GPU):

```bash
pip install -U pip setuptools wheel
pip install \
  "numpy==1.26.4" \
  "pandas==2.2.3" \
  "scipy==1.13.1" \
  "transformers==4.57.6" \
  "accelerate==1.13.0" \
  "peft==0.18.1" \
  "vllm==0.17.1"
```

The per-directory `requirements.txt` files give looser lower bounds if you
prefer newer versions, but the pins above are the known-good combination.

Hardware guidance: a single 24GB+ GPU (A10, L4, A100) is comfortable for the
Qwen3-8B experiments. The paper's runs mostly used a single A100 40GB. The
14B model and DPO training want more headroom (A100 40GB or larger).

### Smoke Test

To verify a working setup end to end in a few minutes, run the main evaluator
on 8 situations:

```bash
cd evaluation
python evaluate.py \
  --base_model Qwen/Qwen3-8B \
  --dataset medium_stakes_validation \
  --num_situations 8 \
  --backend vllm \
  --batch_size 4 \
  --output smoke_vllm.json
```

### Unit Tests

The evaluation package ships its unit tests (no GPU needed):

```bash
cd evaluation
python -m unittest discover -s tests
```

## Paper-Facing Generation Defaults

For the main policy experiments, the paper-facing generation defaults are:

- backend: `vllm`
- temperature: `0.6`
- top-p: `0.95`
- top-k: `20`
- seed: `12345`
- max new tokens: `4096`
- reasoning max tokens: `800`
- batch size: `4`

Qwen models use the shared default system prompt with thinking enabled. Llama
and Gemma runs use no system prompt and the no-think-tag data copies.

## Paper Map

The fastest way to reproduce the main paper results is:

- Main Qwen3-8B method comparison (`tab:qwen8-methods`): run the locked commands in `sft-training/README.md`, `tie-training/README.md`, `dpo-training/README.md`, `reward-model/README.md`, and `steering/README.md`, then evaluate with `evaluation/evaluate.py`.
- Astronomical-stakes cooperation across all five models (`tab:scale-family`): the same commands, run once per base model, and reported for the astronomical-stakes deployment set only.
- Qwen scaling, all four evaluation sets (`tab:qwen-scaling`): use the same method READMEs, swapping the base model to `Qwen/Qwen3-1.7B` or `Qwen/Qwen3-14B`.
- Cross-family results (`tab:cross-family`): use the same method READMEs, swapping the base model to `meta-llama/Llama-3.1-8B-Instruct` or `google/gemma-3-12b-it`. For SFT, tie training, and DPO, use the no-think-tag copies and `--system_prompt ""`.
- Transfer-domain results (`tab:transfer-methods`): run `evaluation/run_transfer_quantity_bundle_eval.py` on the transfer datasets in `evaluation/data/`, or regenerate those CSVs with the scripts in `dataset-generation/`.
- Capability retention (`tab:mmlu-main`): run `evaluation/evaluate_mmlu_redux.py` with the paper-facing deterministic 5-shot settings described in `evaluation/README.md`.
- Comparison against crude heuristics (`tab:heuristics`): run `evaluation/evaluate.py` with `--save_responses`, then `analysis/analyze_model_vs_heuristics.py` on the saved per-situation answers.
- Prize-magnitude sweep (`fig:magnitude-sweep`): build the sweep CSV with `analysis/generate_magnitude_sweep.py`, evaluate it with `evaluation/evaluate.py`, and plot with `analysis/magnitude_sweep_figure.tex`.
- Reward-model held-out evaluation and RewardBench 2: follow `reward-model/README.md`.

## Datasets

The repository ships the exact CSVs used for the paper in `evaluation/data/`.
The dataset-generation scripts are included for transparency and regeneration,
but normal reproduction of the paper tables uses the provided CSVs directly.

## Terminology

Dataset filenames and metrics use the paper's option labels:

- **Cooperate**: the safe/conservative option (lower expected value, less variance).
- **Rebel**: the risky option (higher expected value, more variance). "Rebels" files contain only Rebel-vs-Cooperate choices.
- **Steal**: the option an excessively risk-averse agent would favor. Steals-test situations are constructed so that the target CARA (alpha = 0.01) optimum is a Cooperate option while a much more risk-averse agent (alpha = 0.10) would pick the Steal option, so the **steal rate** measures over-risk-aversion.
- The headline metric is the **cooperate rate**: among validly parsed responses whose chosen option has a behavioral label, the fraction that chose the Cooperate option.
- **CARA** (Constant Absolute Risk Aversion) is the utility function used to label which option is risk-aversion-optimal. CARA-based rates are logged as diagnostics.
- **lin-only** files pair each risk-averse chosen response against a risk-neutral ("linear utility") rejected response.

## Model artifacts

The LoRA adapters, reward heads, and activation-steering direction vectors are
archived separately, because they are large binaries rather than code: five
seeds of each method for each of the five base models.

**<https://huggingface.co/MIT-SERC-risk-averse-AIs/risk-averse-ai-adapter-archive>**

Everything needed to rebuild those artifacts from scratch is also here — the
training data, the trainers, the construction inputs, and the locked
configurations.

## Citation

```bibtex
@misc{zhang2026oodriskaversion,
  title  = {Out-of-Distribution Generalization of Risk Aversion in Language Models},
  author = {Zhang, Kristina and Okoroafor, Junior Chinomso and Maltbie, Benjamin
            and Lin, Andrew and Bokka, Abhitej and Thornley, Elliott},
  year   = {2026},
  note   = {\url{https://github.com/riskaverseAIs/riskaverseAIs}}
}
```

## Licenses

- Code: MIT (see [`LICENSE`](LICENSE)).
- Datasets: CC BY 4.0 (see [`evaluation/DATA_LICENSE.md`](evaluation/DATA_LICENSE.md)).
