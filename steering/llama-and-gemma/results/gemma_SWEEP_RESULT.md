# /home/ubuntu/hf-cache/models--google--gemma-3-12b-it/snapshots/96b6f1eccf38110c56df3a15bffe176da04bfd80 — steering sweep result

alpha = 0 control cooperation: 17.35% (published baseline 17.35%)
Qwen3-8B reference ratio: r = 0.709

| layer | r | alpha | coop | parse | tok-limit | degen | eligible | reasons |
|---:|---:|---:|---:|---:|---:|---:|:--:|---|
| 8 | 0.045 | 263.999 | 28.3% | 97.2% | 2.2% | 1.8% | yes | - |
| 8 | 0.07 | 410.665 | 42.8% | 96.5% | 3.0% | 2.3% | yes | - |
| 8 | 0.1 | 586.664 | 63.9% | 92.8% | 1.5% | 1.2% | no | below_parse_floor |
| 12 | 0.045 | 635.992 | 32.2% | 97.8% | 1.2% | 0.8% | yes | - |
| 12 | 0.07 | 989.321 | 57.5% | 96.2% | 1.5% | 1.2% | yes | - |
| 12 | 0.1 | 1413.316 | 56.1% | 96.5% | 0.0% | 0.0% | yes | - |
| 15 | 0.07 | 2171.983 | 44.3% | 97.8% | 0.0% | 0.0% | yes | - |
| 16 | 0 | 0.000 | 17.3% | 98.0% | 1.5% | 0.5% | yes | - |
| 16 | 0.02 | 691.945 | 36.3% | 98.7% | 0.5% | 0.5% | yes | - |
| 16 | 0.03 | 1037.918 | 44.9% | 98.0% | 0.3% | 0.3% | yes | - |
| 16 | 0.045 | 1556.877 | 59.7% | 98.5% | 0.0% | 0.0% | yes | - |
| 16 | 0.0595 | 2058.538 | 58.5% | 98.5% | 0.3% | 0.3% | yes | - |
| 16 | 0.07 | 2421.809 | 62.7% | 98.7% | 0.3% | 0.3% | yes | - |
| 16 | 0.0826 | 2857.734 | 61.0% | 97.0% | 4.3% | 2.8% | yes | - |
| 16 | 0.1 | 3459.727 | 50.4% | 21.5% | 99.7% | 40.2% | no | below_parse_floor |
| 16 | 0.15 | 5189.590 | 100.0% | 0.2% | 100.0% | 97.8% | no | below_parse_floor |
| 16 | 0.22 | 7611.399 | -- | 0.0% | 100.0% | 8.5% | no | below_parse_floor, no_parsed_answers |
| 16 | 0.32 | 11071.126 | -- | 0.0% | 100.0% | 97.0% | no | below_parse_floor, no_parsed_answers |
| 16 | 0.45 | 15568.771 | -- | 0.0% | 100.0% | 100.0% | no | below_parse_floor, no_parsed_answers |
| 16 | 0.63 | 21796.280 | -- | 0.0% | 100.0% | 0.2% | no | below_parse_floor, no_parsed_answers |
| 16 | 0.85 | 29407.679 | -- | 0.0% | 100.0% | 72.3% | no | below_parse_floor, no_parsed_answers |
| 16 | 1.15 | 39786.860 | -- | 0.0% | 100.0% | 100.0% | no | below_parse_floor, no_parsed_answers |
| 16 | 1.5 | 51895.904 | -- | 0.0% | 100.0% | 100.0% | no | below_parse_floor, no_parsed_answers |
| 17 | 0.07 | 2859.287 | 41.4% | 98.3% | 0.3% | 0.3% | yes | - |
| 20 | 0.045 | 2624.893 | 23.4% | 99.8% | 0.0% | 0.0% | yes | - |
| 20 | 0.07 | 4083.167 | 25.4% | 99.2% | 0.3% | 0.3% | yes | - |
| 20 | 0.1 | 5833.096 | 31.6% | 96.7% | 1.0% | 1.0% | yes | - |
| 24 | 0.045 | 3410.117 | 22.0% | 99.2% | 0.3% | 0.3% | yes | - |
| 24 | 0.07 | 5304.626 | 23.7% | 98.3% | 0.8% | 0.5% | yes | - |
| 24 | 0.1 | 7578.037 | 27.6% | 96.5% | 2.7% | 2.3% | yes | - |

## Outcome

**Promoted:** layer 16, r = 0.07, alpha = 2421.8089
- cooperation 62.67% (+45.32 pp vs the alpha = 0 control, +29.1 SE)
- parse 98.67%

Proceed to the five-seed paper-facing stage only after an independent audit of this table.
