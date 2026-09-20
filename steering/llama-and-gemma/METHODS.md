# Methods — Llama and Gemma steering runs

Written so it can be adapted directly into the paper's methods section or supplementary
material. Everything here describes the September 13–14, 2026 re-sweep and the paper-facing
runs that followed it. Nothing about the Qwen models changed.

## 1. What activation steering does here

A single direction vector is added to the residual stream at one transformer layer, at every
token position, during generation. The direction is unit-normalised, so the size of the
intervention is set entirely by a scalar multiplier.

The direction is built by contrastive activation addition. For each of 200 training
situations, take the residual-stream activation for the prompt plus a risk-averse
chain-of-thought, subtract the activation for the prompt plus a risk-neutral
chain-of-thought, average those differences over situations, and normalise once to unit
length. Construction details follow the paper's existing recipe: all cyclic rotations of the
option order, with option references, outcomes, and answer labels remapped together;
averaging over rotations within a source, then equally across sources; pooling over response
tokens only, with padding excluded.

Steering trains nothing. There are no weights to update and no optimiser.

## 2. Search protocol

Identical to the protocol used for the paper's other sweeps.

- Medium-stakes validation, **first 200 situations in original order**.
- Three construction seeds (12345, 23456, 34567) evaluated at every candidate, pooled —
  600 answers per candidate.
- Decoding seed fixed at 12345.
- Objective: **pooled cooperate count / pooled parsed count.**
- Eligibility: **pooled parse rate ≥ 95%** (≥ 570 of 600). This is the only gate. No
  degeneracy screen, no minimum effect size, no other filter.
- Promotion: among eligible candidates, the highest cooperation rate.

Coordinate search over strength then layer, with step halving, as in the original protocol.

**Why the parse floor matters.** As parse rate falls, cooperation among parsed answers rises
spuriously, because the situations that still produce a parseable answer are not a random
sample. The extreme case observed here: Gemma at r = 0.15 scored 100.0% cooperation on a
0.2% parse rate — one response out of 600. Any table reporting cooperation must report parse
rate beside it.

## 3. Two distinct collapse modes

Worth reporting because they are qualitatively different failures and the paper currently
has no account of either.

- **Llama** degenerates into repetitive looping and then terminates. At r = 0.32, 83.5% of
  responses loop and parse falls to 21%.
- **Gemma** exhausts the 4,096-token budget without ever answering, and usually without
  looping. At r = 0.63, 100% of responses hit the token limit while only 0.2% loop.

So "the vector broke the model" is not one phenomenon. A looping check would catch Llama's
failure and miss Gemma's entirely.

## 4. Selected configurations

| Model | Layer (0-based) | r | Alpha | Medium cooperation at selection |
|:--|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct | 8 | 0.15 | 2.446740245819092 | 34.5% at 99.0% parse |
| Gemma-3-12B-IT | 16 | 0.07 | 2421.8088671875 | 62.67% at 98.67% parse |

**Gemma's optimum is well behaved.** All four refinement neighbours are eligible and score
lower (58.5% at r = 0.0595, 61.0% at r = 0.0826, 44.3% at layer 15, 41.4% at layer 17). It
is a sharp interior peak: layers 20 and 24 are near-inert, cooperating in the low twenties
at 96–99% parse.

**Llama's optimum is not, and this is a declared deviation from the promotion rule.** See §7.

## 5. Paper-facing evaluation

- Five independently **constructed** vectors, paper construction seeds 1–5. These are five
  different direction vectors, not five decoding runs. The decoding seed is pinned at 12345
  for all of them, and for the baselines.
- One baseline per set at alpha = 0, run **through the same hook**, so the baseline exercises
  the identical code path.
- Four sets: medium validation (first 200 situations), high stakes, astronomical stakes, and
  Steals (1,000 each).
- 24 runs and 19,200 answers per model; 48 runs and 38,400 answers in total.
- **Admission gate:** the five medium-validation vector runs must pool ≥ 950 of 1,000 parsed
  before any held-out set is started. Llama pooled 983, Gemma 990. No per-seed veto; no seed
  replaced or discarded.
- Reported: mean and **sample standard deviation across the five per-vector cooperation
  rates**. This is a spread across vectors, not a confidence interval, and should not be
  presented as one.

Generation settings, unchanged from the paper's other steering runs: vLLM backend,
thinking enabled, temperature 0.6, top-p 0.95, top-k 20, max 4,096 new tokens, soft 800-token
reasoning instruction, bfloat16, context 8,192, eager execution while steering, prefix
caching disabled, all-positions application, batch 256.

Llama and Gemma both use the paper's **empty system prompt** convention, with the
thinking-enabled chat template. No Qwen reasoning prompt is used for either.

Pinned model revisions: `meta-llama/Llama-3.1-8B-Instruct` at
`0e9e39f249a16976918f6564b8830bc894c89659`; `google/gemma-3-12b-it` at
`96b6f1eccf38110c56df3a15bffe176da04bfd80`.

Batch 256 was inherited unchanged from the published protocol, where it was selected on a
speed-only benchmark. It was deliberately not retuned: changing it would have made these runs
differ from the published Qwen and Gemma runs in a documented parameter. Different batch
sizes can produce different sampled outputs even at a fixed seed; byte-identical answers
across batch sizes are not claimed.

## 6. The Llama selection departs from the promotion rule

The promotion rule, applied mechanically, selects
Llama layer 10 at r = 0.22: 38.1% cooperation at exactly 570 of 600 parsed, which is the
floor to the answer.

We did not promote it. We promoted layer 8 at r = 0.15 — 34.5% cooperation at 99.0% parse —
which the rule ranks second.

The reason is that the layer-10 point is isolated. Every neighbour fails the 95% floor,
including a *weaker* setting at the same layer (r = 0.187 parses at 94.7%). A point that sits
exactly on an eligibility threshold with no eligible neighbour in any direction is a boundary
artifact of where the grid happened to land, not a configuration anyone should report as the
method's operating point. Promoting it would also have put the headline number on a setting
whose parse rate could fall below the floor under resampling.

This is a judgement call that departs from the stated rule, and it is recorded here so a
reader can disagree with it. Llama's reported number is therefore somewhat conservative: the
mechanical rule would have produced a larger effect.

Gemma required no such call.

## 7. Selection bias in the search estimates

The search estimate at a promoted configuration is optimistically biased, because the
configuration was chosen using those same three search seeds.

| Model | Search estimate (3 search seeds) | Five fresh vectors | Optimism |
|:--|---:|---:|---:|
| Llama-3.1-8B | 34.5% | 31.65% | 2.85 pp |
| Gemma-3-12B | 62.67% | 62.52% | 0.15 pp |

The bias is a property of the search landscape rather than of the method: Llama's optimum is
narrow, so the search seeds could be lucky there; Gemma's is broad, so they could not. The
fresh-vector numbers are the ones reported. This is precisely what constructing new vectors
after selection is for, and it is worth a sentence in the methods.

## 8. What these runs do not establish

- They do not show the vector encodes risk aversion specifically rather than some correlated
  feature. No random-direction control and no length-matched control was run for Llama or
  Gemma.
- They do not show the model reasons about risk *well*. Llama's sampled transcripts invent a
  non-standard utility function and contain incorrect intermediate values. What the runs show
  is that the vector changes which option is chosen, in coherent, on-topic text.
- They do not establish a global optimum. The search is local, and the protocol says so.
- No capability-retention check (MMLU or similar) was run at these settings for
  either model. The published Qwen3-8B steering result cost about 24 points of MMLU accuracy,
  so this is a real gap.
