# Dataset audit

502 sequences x 30 frames x 126 features, 14 classes.

| class | n | frames w/ hand | mostly-two-hand samples | samples <50% hand frames |
|---|---|---|---|---|
| back | 40 | 0.93 | 1 | 0 |
| close | 40 | 0.95 | 40 | 0 |
| down | 40 | 0.94 | 1 | 2 |
| front | 40 | 0.94 | 0 | 0 |
| hello | 31 | 0.94 | 4 | 0 |
| home | 30 | 0.96 | 29 | 0 |
| left | 40 | 0.96 | 36 | 1 |
| no | 30 | 1.00 | 6 | 0 |
| open | 40 | 0.85 | 36 | 1 |
| pick | 31 | 0.98 | 0 | 0 |
| place | 30 | 1.00 | 29 | 0 |
| right | 40 | 0.98 | 39 | 0 |
| up | 40 | 0.95 | 2 | 1 |
| yes | 30 | 1.00 | 13 | 0 |

## Findings
- Exact duplicate sequences: 0
- Class imbalance (max/min): 1.33
- Samples with <50% hand frames (likely tracking failures, candidates to drop/re-record): 5
- Normalization invariant is enforced by `python -m scripts.fix_landmarks --verify`.
- All samples come from ONE signer/setup: random-split accuracy is optimistic. Use the `block` split and see docs/EVALUATION.md.
