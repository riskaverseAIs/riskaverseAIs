# /home/ubuntu/hf-cache/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 — steering sweep result

alpha = 0 control cooperation: 15.90% (published baseline 15.90%)
Qwen3-8B reference ratio: r = 0.709

| layer | r | alpha | coop | parse | tok-limit | degen | eligible | reasons |
|---:|---:|---:|---:|---:|---:|---:|:--:|---|
| 6 | 0.15 | 2.273 | 39.0% | 83.3% | 0.0% | 0.0% | no | below_parse_floor |
| 7 | 0.15 | 2.376 | 28.8% | 97.2% | 0.2% | 0.2% | yes | - |
| 8 | 0.1 | 1.631 | 24.3% | 98.2% | 0.0% | 0.0% | yes | - |
| 8 | 0.15 | 2.447 | 34.5% | 99.0% | 0.3% | 0.0% | yes | - |
| 8 | 0.22 | 3.589 | 42.2% | 87.3% | 6.0% | 3.8% | no | below_parse_floor |
| 9 | 0.22 | 3.691 | 40.6% | 81.7% | 7.8% | 6.2% | no | below_parse_floor |
| 10 | 0.1 | 1.711 | 22.9% | 97.5% | 0.3% | 0.0% | yes | - |
| 10 | 0.15 | 2.566 | 31.0% | 97.2% | 0.3% | 0.2% | yes | - |
| 10 | 0.187 | 3.199 | 33.6% | 94.7% | 1.8% | 1.5% | no | below_parse_floor |
| 10 | 0.22 | 3.763 | 38.1% | 95.0% | 3.2% | 2.3% | yes | - |
| 10 | 0.2596 | 4.441 | 47.0% | 79.0% | 18.7% | 11.5% | no | below_parse_floor |
| 11 | 0.22 | 3.808 | 29.1% | 87.0% | 13.7% | 8.7% | no | below_parse_floor |
| 12 | 0 | 0.000 | 15.9% | 97.5% | 0.0% | 0.0% | yes | - |
| 12 | 0.02 | 0.361 | 17.8% | 96.5% | 0.0% | 0.0% | yes | - |
| 12 | 0.03 | 0.542 | 18.5% | 96.5% | 0.0% | 0.0% | yes | - |
| 12 | 0.045 | 0.813 | 17.8% | 95.5% | 0.0% | 0.0% | yes | - |
| 12 | 0.07 | 1.265 | 16.9% | 94.8% | 0.0% | 0.0% | no | below_parse_floor |
| 12 | 0.1 | 1.807 | 21.6% | 95.5% | 0.3% | 0.0% | yes | - |
| 12 | 0.15 | 2.711 | 28.6% | 96.0% | 1.2% | 1.0% | yes | - |
| 12 | 0.22 | 3.976 | 34.8% | 55.0% | 44.5% | 23.8% | no | below_parse_floor |
| 12 | 0.32 | 5.783 | 47.6% | 21.0% | 94.0% | 83.5% | no | below_parse_floor |
| 12 | 0.45 | 8.133 | -- | 0.0% | 100.0% | 99.3% | no | below_parse_floor, no_parsed_answers |
| 12 | 0.63 | 11.386 | -- | 0.0% | 100.0% | 100.0% | no | below_parse_floor, no_parsed_answers |
| 12 | 0.85 | 15.362 | -- | 0.0% | 100.0% | 100.0% | no | below_parse_floor, no_parsed_answers |
| 12 | 1.15 | 20.784 | -- | 0.0% | 100.0% | 100.0% | no | below_parse_floor, no_parsed_answers |
| 12 | 1.5 | 27.110 | -- | 0.0% | 100.0% | 100.0% | no | below_parse_floor, no_parsed_answers |
| 14 | 0.1 | 1.882 | 22.1% | 96.3% | 0.3% | 0.0% | yes | - |
| 14 | 0.15 | 2.823 | 19.9% | 93.2% | 2.5% | 1.3% | no | below_parse_floor |
| 14 | 0.22 | 4.141 | 30.2% | 77.8% | 19.8% | 11.2% | no | below_parse_floor |
| 16 | 0.1 | 2.089 | 23.3% | 97.8% | 0.3% | 0.3% | yes | - |
| 16 | 0.15 | 3.133 | 23.2% | 96.2% | 0.7% | 0.2% | yes | - |
| 16 | 0.22 | 4.595 | 26.0% | 91.7% | 3.3% | 2.5% | no | below_parse_floor |

## Outcome

**Promoted:** layer 10, r = 0.22, alpha = 3.7632
- cooperation 38.07% (+22.17 pp vs the alpha = 0 control, +14.5 SE)
- parse 95.00%

Proceed to the five-seed paper-facing stage only after an independent audit of this table.
