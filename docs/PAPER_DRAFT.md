# Sign Commands for Robot Arms: A Measurement Study of Feature-Space Mismatch, Split Leakage and Idle-Time False Activations

**DRAFT v0.1, preliminary. Every number below is from this repository (`reports/`, `scripts/`). Items marked `[TODO]` are NOT yet measured. Nothing here may be claimed until it is filled in.**
Authors: [TODO]. Code: github.com/devxsanj/SignVLA.

> **Working title note.** Two arXiv papers already use the name "SignVLA" (2602.22514, 2606.20857; see `docs/NOVELTY_ASSESSMENT.md`). Rename before submission.

## Abstract
Sign-language-driven robot control is usually reported as a classification accuracy on a fixed vocabulary. We argue that for a *command* channel the relevant quantities are false activations while the operator is idle, rejection of unknown input, and latency, and that accuracy on a single-signer dataset is easily inflated by pipeline errors and split leakage. Using a 14-command Indian Sign Language (ISL) vocabulary, a MediaPipe-landmark + Bi-GRU encoder and a language-agnostic concept layer feeding a simulated 4-DOF arm, we report (i) a silent train/inference feature-space mismatch that affected 8 of 14 classes, (ii) a 12-point accuracy gap between random and recorded-later splits (90.1% vs 78.0%), and (iii) that adding a learned *rest* class and idle recordings changes, but does not remove, false activations (held-out accuracy 73.0%, 78.8%, 63.0% after one, two and three idle sessions, as the held-out idle data became harder). We do not evaluate a VLA, other signers, a second sign language, or hardware; these are future work.

## 1. Introduction
Camera-based sign or gesture control of low-cost arms is well established (Section 2). Its evaluation rarely covers what an operator actually experiences: the robot moving when they did not sign. We treat sign as a *command channel* and ask: how reliable is the channel when idle, and which experimental mistakes make it look better than it is?

Contributions (as far as evidenced):
1. A documented, reproducible failure case: a mixed raw/normalized feature space that let the classifier separate classes by feature scale, plus the fix and a verified invariant (`scripts/fix_landmarks.py`).
2. Evidence that random splits of back-to-back recordings overstate accuracy (Section 5.1).
3. An idle ("rest") class and rejection gates (confidence, margin, hand-presence, debounce), with measured effect and limits (Section 5.2).
4. A concept layer separating sign vocabulary from robot primitives, and a headless simulator check (Section 4.3). Cross-language interchangeability is **not yet measured** `[TODO]`.

## 2. Related work
See `docs/NOVELTY_ASSESSMENT.md` for the verified/unverified reference table. Closest work: SignVLA (Tan et al. 2602.22514; Bai et al. 2606.20857), sign-to-text-to-VLA pipelines; gesture-conditioned VLAs (GesVLA 2605.22812, GIVE 2606.13435; pointing, not sign); cross-lingual sign representations (Wei & Chen, ICCV 2023; CISLR; SignCLIP). Abstract-level comparisons only; full texts must be read before submission `[TODO]`.

## 3. Data
- 15 classes: 14 ISL commands (back, close, down, front, hello, home, left, no, open, pick, place, right, up, yes; 30-40 sequences each, 502 total) + `rest`.
- Each sequence: 30 frames x 126 features (two hands x 21 landmarks x xyz), ~1 s at 30 fps, mirrored camera, one signer, one setup.
- `rest`: continuous idle footage sliced into 30-frame windows (stride 15) over three sessions (132, 170, 200 windows; hand visible in 35%, 50%, 94% of frames). Windows overlap by 50%.
- Limitations: single signer, no signer- or session-independent test, overlapping rest windows, mixed one-/two-handed signing for yes/no/hello (`reports/dataset_audit.md`).

## 4. Method
### 4.1 Features
Per hand: wrist-translated, divided by wrist-to-middle-MCP distance. Hand slots: lone hand -> slot 0; two hands -> image-left first (independent of MediaPipe's handedness label). One shared function is used for recording, repair, training and live inference (`signvla/perception/features.py`); a checkpoint carries a `feature_version` and old ones are refused.
### 4.2 Encoder and gates
Linear+LayerNorm -> 2-layer Bi-GRU (hidden 128) -> temporal mean -> MLP; AdamW, cosine schedule, Gaussian jitter on present landmarks, model selection on validation only, test scored once. A window is accepted as a command only if the top class is not `rest`, >=60% of frames contain a hand, top-1 probability >=0.75 and top-1 minus top-2 >=0.30; then 3 consecutive agreeing windows and a 1.2 s cooldown. Thresholds were set by hand, not tuned `[TODO: sweep on validation, report ROC]`.
### 4.3 Concept layer and simulator
sign (language, label) -> `Concept` -> primitive; the robot process receives only concept names over UDP. 14 primitives run headless on the RoArm-M2-Pro MJCF; all reach the final waypoint within 0.05 rad. This is joint-level reach, **not task success** `[TODO: object-level success]`.

## 5. Results (all preliminary, single signer)
### 5.1 Feature mismatch and split leakage (14 classes, before `rest`)
| Split | Test accuracy |
|---|---|
| Random stratified | 90.1% |
| Block (latest recordings per class held out) | 78.0% |

Before the fix, 319/502 samples were raw coordinates (8 classes) and 183 normalized (6 classes); live inference produces normalized features only. We did not measure pre-fix live accuracy `[TODO if the old checkpoint is retrievable: it is in git history]`.
### 5.2 Effect of the rest class (block split, held-out idle windows come from the last part of the newest session)
| Idle sessions | Sequences | Top-1 | Rest recall | Accepted-but-wrong* |
|---|---|---|---|---|
| 1 | 634 | 73.0% | 38% | 19.8% |
| 2 | 804 | 78.8% | 78% | 13.1% |
| 3 | 1004 | 63.0% | 51% | 17.5% |
| 3, class-weighted loss | 1004 | 55.5% | 31% | 31.5% |

*share of all test windows that pass every gate with a wrong label. Rows are **not directly comparable**: the test set changes with each session. Session 3 held-out windows are the hardest idle behavior (open hand drifting toward the camera): 20 of 100 were classified `back`. Class weighting made results worse, so it was reverted. Weak classes throughout: `no` (17%), `open` (25-50%).
### 5.3 Live run (unlabeled)
One 800 s live session produced 65 fired commands, mean inference 24 ms. Ground truth was not logged, so no accuracy or false-activation rate can be stated. **`[TODO] the key experiment`**: a labeled protocol (e.g. 10 min idle + 20 attempts per class), reporting false activations per minute and end-to-end latency percentiles.

## 6. Discussion and honest limits
- The rest-vs-sign ambiguity (`back` vs. an open drifting hand) is inherent to single-window landmark shape; better data balance did not fix it. Candidate remedies: a deliberate start pose or dwell time, motion features, re-designed signs `[TODO]`.
- The gap between validation (85%) and block-test (63%) indicates strong drift between sessions even for one signer.
- **No VLA is implemented.** The architecture proposes an OpenVLA interface; there is no experiment. Any VLA claim requires an ablation (concept layer alone vs. + VLA) `[TODO]`.
- No hardware or sim-to-real experiment. The serial controller in `experiments/hardware_untested/` is unverified.

## 7. Minimum work before submission
1. Labeled live protocol (5.3) with false activations per minute, >=2 operators.
2. Signer-independent evaluation (>=5 signers; `data/isl/isl40` clips can seed a second source; check licences).
3. Baselines: kNN/DTW, TCN, Transformer; a validation-tuned gate with ROC.
4. Second sign language into the concept layer, and measure interchangeability.
5. Task-level MuJoCo success; then hardware with verified commands.
6. Only then: OpenVLA ablation, if it changes a measured number.
7. Read the full texts of the two SignVLA papers; rename the project.

## Reproduce
`pip install -r requirements.txt && pip install -e . && python -m scripts.build_dataset && python -m scripts.train && python -m scripts.evaluate` (see README).
