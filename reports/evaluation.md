# Evaluation (split = block, n_test = 100)

- Top-1 accuracy: **78.0%**  (val at checkpoint: 94.6%)
- Gates: conf >= 0.75, margin >= 0.3, hand frames >= 0.6
- Test windows accepted: 83.0%  | accepted AND correct: 71.0%  | accepted but WRONG (false command): 12.0%
- No-hand windows rejected: 100%
- Spliced transition windows rejected (proxy for ambiguous input): 69%  (n=282)

## Per-class recall

| class | n | recall | most confused with |
|---|---|---|---|
| back | 8 | 100% | - |
| close | 8 | 88% | open (1) |
| down | 8 | 100% | - |
| front | 8 | 88% | no (1) |
| hello | 6 | 67% | place (1) |
| home | 6 | 100% | - |
| left | 8 | 88% | down (1) |
| no | 6 | 17% | yes (2) |
| open | 8 | 38% | close (2) |
| pick | 6 | 67% | hello (2) |
| place | 6 | 100% | - |
| right | 8 | 88% | front (1) |
| up | 8 | 75% | back (2) |
| yes | 6 | 67% | back (2) |

## Confusion matrix (rows = true, cols = predicted)

```
         back clos down fron hell home left   no open pick plac righ   up  yes
back        8    0    0    0    0    0    0    0    0    0    0    0    0    0
close       0    7    0    0    0    0    0    0    1    0    0    0    0    0
down        0    0    8    0    0    0    0    0    0    0    0    0    0    0
front       0    0    0    7    0    0    0    1    0    0    0    0    0    0
hello       0    0    0    0    4    0    0    0    0    0    1    0    0    1
home        0    0    0    0    0    6    0    0    0    0    0    0    0    0
left        0    0    1    0    0    0    7    0    0    0    0    0    0    0
no          0    1    1    0    0    1    0    1    0    0    0    0    0    2
open        0    2    0    0    1    1    0    0    3    0    1    0    0    0
pick        0    0    0    0    2    0    0    0    0    4    0    0    0    0
place       0    0    0    0    0    0    0    0    0    0    6    0    0    0
right       0    0    0    1    0    0    0    0    0    0    0    7    0    0
up          2    0    0    0    0    0    0    0    0    0    0    0    6    0
yes         2    0    0    0    0    0    0    0    0    0    0    0    0    4
```

Caveat: single signer, single setup. See docs/EVALUATION.md for what is NOT yet measured.
