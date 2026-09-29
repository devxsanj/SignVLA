# SignVLA — language-agnostic sign → robot-action interface

```
camera → MediaPipe hands → 30×126 landmarks → Bi-GRU sign encoder → confidence/margin/hand gates
       → debounce → SEMANTIC CONCEPT (language-agnostic) → robot primitive → MuJoCo (→ hardware, TODO)
```
Read `docs/NOVELTY_ASSESSMENT.md` **first**: two arXiv papers named "SignVLA" already exist, so the name and any "first" claim need rethinking.

## Setup (Python 3.10–3.12)
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
pytest -q            # 22 tests, no camera needed
```

## The commands you need
```bash
python -m scripts.fix_landmarks --verify   # data sanity (canonical features)
python -m scripts.build_dataset            # data/landmarks -> data/dataset.npz
python -m scripts.train                    # train on the held-out "block" split -> models/
python -m scripts.evaluate                 # -> reports/evaluation.md
mjpython -m scripts.run_sim                # terminal 1: MuJoCo viewer (macOS needs mjpython)
python -m scripts.run_live                 # terminal 2: camera -> commands
```
Record new signs: `python -m scripts.record --gestures pick --samples 40`, then rebuild + retrain.

## Layout
| path | what |
|---|---|
| `signvla/perception/features.py` | **the one definition of the feature space** — never re-implement it |
| `signvla/encoder/` | dataset + splits, Bi-GRU, training |
| `signvla/live/` | gated recognizer, debouncer, camera loop |
| `signvla/semantics/concepts.py` | sign (language, label) → `Concept` → robot primitive. Add ASL here |
| `signvla/robot/roarm_sim.py` | primitives, trajectory player, viewer, headless runner |
| `scripts/` | runnable entry points |
| `data/landmarks/` | the 502 samples; videos, `isl/` external clips, npz are git-ignored |
| `reports/` | generated audit + evaluation |
| `experiments/` | old scripts kept for reference |

## What was fixed in v0.2 (this explains the "weak real-time" problem)
1. **Train/live feature mismatch (root cause).** 8 classes (back, close, down, front, left, open, right, up) were recorded
   with a script saving *raw* MediaPipe coordinates; the other 6 were normalized; live inference always normalizes. The
   model could separate classes by "raw vs normalized". All 502 samples are now converted to one canonical space
   (`scripts/fix_landmarks.py`, invariants verified; originals in `data/_legacy_landmarks_backup/`). No re-recording needed.
2. Hand slot no longer depends on MediaPipe's Left/Right label (lone hand → slot 0, two hands → left-most first).
3. Model selection was done on the same split reported as accuracy. Now train/val/test; test scored once.
4. Added a **block split** (later recordings held out): a random split leaks near-duplicate takes.
5. Live gating: no-hand, confidence, top-1/top-2 margin, debounce, cooldown (before, the model ran on empty frames).
6. `src/gesture_to_sim.py` was a pasted shell heredoc (SyntaxError) — rewritten. `mediapipe` pinned to 0.10.21.
7. Direct gesture→action table replaced by the semantic layer; the simulator only receives concept names.

## Honest current numbers (`reports/evaluation.md`)
15 classes incl. `rest` (idle class, never commands the robot; record more with `scripts.record_rest`).
Held-out later recordings: **78.8%** top-1, rest recall 78%, 13% of test windows pass the gates with a wrong label.
Weak: `no` 17%, `open` 25%, `hello` 67%. Random-split accuracy is higher (leakage). Single signer, single setup:
no generalization claim is possible yet. See `docs/EVALUATION.md`.
