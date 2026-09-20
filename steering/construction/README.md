# Counterbalanced construction files

The five inputs used to build the paper's steering directions, one per construction seed.

Each file starts from the 600-row lin-only low-stakes chain-of-thought set
(`../../evaluation/data/2026_03_22_low_stakes_training_set_600_situations_with_CoTs_lin_only.csv`),
samples 200 source situations, and then emits one row per cyclic rotation of the option
order. Option labels and every reference to them inside both chains of thought are remapped
to match the new display order, so a rotation changes only where each option appears.

| | |
|---|---|
| rows per file | 587 |
| source situations | 200 |
| orderings per source | 2 (80 sources), 3 (67), 4 (39), 5 (14) |

Averaging over the rotations within a source removes any effect of where an option happens to
be displayed. Because sources have different numbers of options, they contribute different
numbers of rows, so the average has to be taken within a source first and then across
sources — which is what `--source_column situation_id` does in
[`../build_steering_direction.py`](../build_steering_direction.py).

## Audits

`CONSTRUCTION_BUILD_SUMMARY.json` records, per seed, the source file and its SHA-256, the
output SHA-256, the selection quotas, and four checks that each file passed when it was built:

- every semantic option and both targets occupy every display position exactly once within
  each source;
- prompt option bodies are moved without modification;
- applying the inverse relabelling recovers each original response byte for byte;
- every final JSON answer equals the remapped expected label.

## Columns

`situation_id` identifies the source situation and is what groups the rotations.
`prompt_text`, `chosen_full`, and `rejected_full` are what the builder reads: the prompt, the
risk-averse chain of thought, and the risk-neutral one. The remaining columns carry the
option labels under each utility function and the answer-validation fields.
