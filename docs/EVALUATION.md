# Evaluation status and next steps (maps to the stages of the research architecture)

| Stage | State | Next action |
|---|---|---|
| 1 Code audit | done (README "What was fixed") | — |
| 2 Data verification | done: `reports/dataset_audit.md` | drop/re-record 5 samples with <50% hand frames; `yes/no/hello` mix one- and two-hand signing — pick one convention |
| 3 Retrain | done once, on repaired features | don't retrain again before stage 4 |
| 4 Real-time validation | **not done** (no camera in this session) | `run_live --no-send --log`, 20 attempts per class, count false fires while idle for 10 min |
| 5 Semantic layer | done (`semantics/concepts.py`) | ablation direct vs concept path is trivial until a 2nd language exists |
| 6 Multi-sign | stub (`LEXICONS["asl"]`) | pick clips from `data/isl/isl40` (check licences) |
| 7 VLA experiment | not started | only if it changes a measured number |
| 8 MuJoCo | `run_sim --headless`: joint-level reach only | add task-level success (object moved) before claiming manipulation |
| 9 Hardware | not started | verify Waveshare JSON commands first |
| 10 Ablations | not started | kNN/DTW + TCN + Transformer baselines; leave-one-signer-out |

## Known measurement gaps
- Live uses a sliding 30-frame window; training clips were countdown-aligned. Probable cause of remaining live errors; test with random-shift training crops.
- `evaluate.py` rejection numbers are proxies (zero windows, spliced windows). Real idle footage is required.
- Wrist-centred features drop absolute hand position, so position-defined signs (left/right/up/down) rely on hand shape/orientation; test wrist-trajectory features (needs re-recording the 6 normalized classes).
- Hand slots follow image position, so a spurious second hand can swap slot order.
