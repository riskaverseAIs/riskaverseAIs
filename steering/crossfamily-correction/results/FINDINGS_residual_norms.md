# Measured residual-stream norms — the number the September runs never recorded

**13 September 2026, measured on an H100 from a clone of the September volume.**

The steering hook passed every check on both Llama and Gemma, so the intervention was being
applied correctly. What was wrong is the *scale* it was applied at.

## The measurements

Steering directions are unit-normalised, so the strength `alpha` is an absolute quantity
added to a residual stream whose own size differs enormously between models. The ratio that
matters is `r = alpha / mean_residual_norm_at_layer`.

| Model | Layer | Mean residual norm | Alpha for Qwen's r = 0.709 |
|---|---:|---:|---:|
| Qwen3-8B (recorded Sept) | 12 | 45.12 | 32 ← the working setting |
| Llama-3.1-8B-Instruct | 8 | 16.31 | 11.57 |
| Llama-3.1-8B-Instruct | 10 | 17.11 | 12.13 |
| Llama-3.1-8B-Instruct | 12 | 18.07 | 12.82 |
| Llama-3.1-8B-Instruct | 14 | 18.82 | 13.35 |
| Llama-3.1-8B-Instruct | 16 | 20.89 | 14.81 |
| **Gemma-3-12B-IT** | **16** | **40,361.56** | **28,625** |

Gemma's residual stream is about **900 times larger than Qwen's** and **2,200 times larger
than Llama's**. Gemma-3 scales its embeddings by sqrt(hidden_size) and carries very large
activations; nothing here is anomalous for that architecture. It is simply a fact nobody
measured, because the build summary recording `mean_residual_norm_at_layer` was kept only
for Qwen3-8B.

## What this means for Gemma

The September Gemma search swept alpha from 0 to 48. In ratio terms:

| Sept alpha | r | as % of the residual stream |
|---:|---:|---:|
| 8 | 0.00020 | 0.02% |
| 16 (selected) | 0.00040 | 0.04% |
| 48 (largest ever tried) | 0.00119 | **0.12%** |
| Qwen's working point | 0.709 | 71% |

The largest perturbation ever applied to Gemma was **0.12% of its residual stream**, against
Qwen's 71%. That is roughly **600 times weaker** than even the smallest point on the new
grid, and about 24,000 times weaker than the Qwen-equivalent setting.

This explains the September Gemma result exactly: every candidate sat between 16.8% and
20.2% cooperation, the alpha = 0 control sat inside that range, and the parse rate never
degraded anywhere — not even at layer 0. The vector was doing essentially nothing. The
reported "Gemma does not transfer" finding carries no information about Gemma.

## What this means for Llama — a genuine refinement

Llama is a different story, and the audit's original framing needs correcting here.

At layer 12 the norm is 18.07, so the September settings were:

| Sept alpha | r | outcome |
|---:|---:|---|
| 0.5 | 0.028 | no effect, parse 98.3% |
| 0.074 (selected) | 0.0041 | no effect, parse 97.7% |
| 16 | 0.885 | **total collapse**, parse 0%, 100% token-limit |

So Llama's collapse at alpha 16 happened at **r = 0.885, above Qwen's working ratio of
0.709** — not far below it. Scale alone therefore does *not* explain the Llama null the way
it explains the Gemma null. Llama really is more fragile than Qwen at comparable relative
strength.

What remains true, and is the reason the sweep is worth running, is that the entire band
from r = 0.028 to r = 0.885 went untested — including the Qwen-equivalent point at
r = 0.709 (alpha ≈ 12.8). The new grid covers that band densely.

## Hook verification

Both models passed every check, using the real `ResidualSteeringHook` and
`get_decoder_layers` imported from `evaluate.py`:

- Llama resolves to `model.layers`, 32 blocks, matching config.
- Gemma resolves to `model.language_model.layers`, 48 blocks, matching config — the nested
  multimodal path is handled correctly, so the layer-indexing worry was unfounded.
- alpha = 0 is a bitwise no-op on both; the hooked block's output differs by exactly
  `alpha x direction`; all positions are steered; and logits move measurably, so the hook is
  not a silent no-op.

## Bottom line for the paper

The Gemma null must be withdrawn — it measures nothing. The Llama null is better founded
than I first said, but still rests on an untested band that contains the setting most
directly comparable to Qwen's.

---

## Addendum, 13 September 2026 — the norm depends on the measuring prompt

The layer-16 Gemma figure above (40,361.56) came from `hook_audit.py`, which measures
on a short one-line gamble. Re-measuring with `measure_norms.py`, which uses a longer
benchmark-shaped scenario, gives **34,597.27** at the same layer — about 17% lower.
This is not nondeterminism: re-measuring Llama with an identical script and prompt
reproduced all five previously recorded layers bit-for-bit.

Mean residual norm is an average over the tokens of whatever text is fed in, so it is
a property of the model *and* the prompt, not a constant of the model. Two reasonable
prompts differ by 17% on Gemma.

What this does and does not affect:

- **Within a sweep: nothing.** Every candidate in a given sweep converts r to alpha
  through the same measured norm, so comparisons between candidates — which layer,
  which strength, which is promoted — are exact.
- **Across models: nothing that matters here.** Gemma's residual stream is roughly a
  thousand times Llama's. A 17% definitional wobble does not touch a conclusion that
  rests on three orders of magnitude.
- **The Qwen anchor is approximate.** The reference value 45.12 was recorded by the
  September pipeline under a third convention again, so "Qwen was steered at r = 0.709"
  is a landmark, not a precise figure. It should be described that way.

The rule this implies for the paper: report r together with how the norm was measured.
Recording the norm but not the convention would be a milder version of the original
failure, which was recording no norm at all.

The sweeps in this directory all use `measure_norms.py` values, listed in each sweep's
`RESIDUAL_NORMS.json`.
