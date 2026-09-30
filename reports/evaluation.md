# Evaluation (split = block, n_test = 200)

- Top-1 accuracy: **63.0%**  (val at checkpoint: 85.2%)
- Gates: conf >= 0.75, margin >= 0.3, hand frames >= 0.6
- Test windows accepted: 50.0%  | accepted AND correct: 32.5%  | accepted but WRONG (false command): 17.5%
- No-hand windows rejected: 100%
- Spliced transition windows rejected (proxy for ambiguous input): 72%  (n=208)

## Per-class recall

| class | n | recall | most confused with |
|---|---|---|---|
| back | 8 | 100% | - |
| close | 8 | 75% | open (1) |
| down | 8 | 100% | - |
| front | 8 | 100% | - |
| hello | 6 | 83% | place (1) |
| home | 6 | 100% | - |
| left | 8 | 88% | rest (1) |
| no | 6 | 17% | rest (3) |
| open | 8 | 50% | rest (3) |
| pick | 6 | 0% | hello (3) |
| place | 6 | 100% | - |
| rest | 100 | 51% | back (20) |
| right | 8 | 75% | front (1) |
| up | 8 | 75% | rest (2) |
| yes | 6 | 67% | back (2) |

## Confusion matrix (rows = true, cols = predicted)

```
         back clos down fron hell home left   no open pick plac rest righ   up  yes
back        8    0    0    0    0    0    0    0    0    0    0    0    0    0    0
close       0    6    0    0    0    0    0    0    1    0    0    1    0    0    0
down        0    0    8    0    0    0    0    0    0    0    0    0    0    0    0
front       0    0    0    8    0    0    0    0    0    0    0    0    0    0    0
hello       0    0    0    0    5    0    0    0    0    0    1    0    0    0    0
home        0    0    0    0    0    6    0    0    0    0    0    0    0    0    0
left        0    0    0    0    0    0    7    0    0    0    0    1    0    0    0
no          0    0    0    0    1    0    0    1    0    0    0    3    0    0    1
open        0    0    0    0    0    1    0    0    4    0    0    3    0    0    0
pick        0    0    0    0    3    0    0    0    0    0    0    3    0    0    0
place       0    0    0    0    0    0    0    0    0    0    6    0    0    0    0
rest       20    6    3    1   14    0    0    3    0    0    0   51    2    0    0
right       0    0    0    1    0    0    0    0    0    0    1    0    6    0    0
up          0    0    0    0    0    0    0    0    0    0    0    2    0    6    0
yes         2    0    0    0    0    0    0    0    0    0    0    0    0    0    4
```

Caveat: single signer, single setup. See docs/EVALUATION.md for what is NOT yet measured.
