# Evaluation (split = block, n_test = 160)

- Top-1 accuracy: **78.8%**  (val at checkpoint: 90.8%)
- Gates: conf >= 0.75, margin >= 0.3, hand frames >= 0.6
- Test windows accepted: 59.4%  | accepted AND correct: 46.2%  | accepted but WRONG (false command): 13.1%
- No-hand windows rejected: 100%
- Spliced transition windows rejected (proxy for ambiguous input): 76%  (n=233)

## Per-class recall

| class | n | recall | most confused with |
|---|---|---|---|
| back | 8 | 88% | yes (1) |
| close | 8 | 88% | open (1) |
| down | 8 | 100% | - |
| front | 8 | 88% | hello (1) |
| hello | 6 | 67% | place (1) |
| home | 6 | 100% | - |
| left | 8 | 88% | rest (1) |
| no | 6 | 17% | yes (2) |
| open | 8 | 25% | home (3) |
| pick | 6 | 83% | hello (1) |
| place | 6 | 100% | - |
| rest | 60 | 78% | down (5) |
| right | 8 | 88% | front (1) |
| up | 8 | 88% | rest (1) |
| yes | 6 | 83% | back (1) |

## Confusion matrix (rows = true, cols = predicted)

```
         back clos down fron hell home left   no open pick plac rest righ   up  yes
back        7    0    0    0    0    0    0    0    0    0    0    0    0    0    1
close       0    7    0    0    0    0    0    0    1    0    0    0    0    0    0
down        0    0    8    0    0    0    0    0    0    0    0    0    0    0    0
front       0    0    0    7    1    0    0    0    0    0    0    0    0    0    0
hello       0    0    0    0    4    0    0    0    0    0    1    0    0    0    1
home        0    0    0    0    0    6    0    0    0    0    0    0    0    0    0
left        0    0    0    0    0    0    7    0    0    0    0    1    0    0    0
no          0    0    0    0    0    0    1    1    0    0    0    1    0    1    2
open        0    0    0    0    0    3    0    0    2    1    0    2    0    0    0
pick        0    0    0    0    1    0    0    0    0    5    0    0    0    0    0
place       0    0    0    0    0    0    0    0    0    0    6    0    0    0    0
rest        3    0    5    2    0    0    2    1    0    0    0   47    0    0    0
right       0    0    0    1    0    0    0    0    0    0    0    0    7    0    0
up          0    0    0    0    0    0    0    0    0    0    0    1    0    7    0
yes         1    0    0    0    0    0    0    0    0    0    0    0    0    0    5
```

Caveat: single signer, single setup. See docs/EVALUATION.md for what is NOT yet measured.
