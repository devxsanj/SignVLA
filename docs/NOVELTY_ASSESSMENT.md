# SignVLA: Novelty / Prior-Art Assessment

Date of check: 2026-09-29. Method: web search plus fetching arXiv abstract pages. Only abstracts/metadata were read; full papers were NOT read, so "how it differs" claims are based on abstracts and must be re-checked against full texts before submission. Items marked **[UNVERIFIED]** were seen only in a search snippet or cited from memory.

## 0. Headline findings (read first)

1. **Name and concept collision.** Two arXiv papers already titled "SignVLA" exist, both from the same author group, on the same idea (sign language to landmarks to LSTM to text/semantic instruction to VLA to robot arm):
   - *SignVLA: A Gloss-Free Vision-Language-Action Framework for Real-Time Sign Language-Guided Robotic Manipulation* (arXiv 2602.22514, Feb 2026), which claims to be "the first sign language-driven VLA framework". It covers finger-spelling.
   - *SignVLA: Real-Time Sign Language-Guided Robotic Manipulation via Attention LSTM and VLA Models* (arXiv 2606.20857, Jun 2026): MediaPipe-style hand landmarks, attention-LSTM, "modular sign-to-text interface", Franka Panda.
   Keep the name and a reviewer will assume derivative work or confusion. **Rename the project** (or explicitly position against these papers). The claim "first sign-driven VLA" is unavailable.
2. "Landmarks + recurrent net + sign-to-command + robot arm" is not novel. It is a dozen student projects and several papers.
3. Language-agnostic sign recognition and cross-sign-language transfer are established research areas (multilingual ISLR, SignCLIP, SONAR-SLT, CISLR's ASL-to-ISL transfer).
4. What remains open is narrow: a *safety/evaluation-centred* contribution (false activation, unknown rejection, latency, sim-to-real for sign-commanded arms) and a *shared concept interface with measured cross-language interchangeability*. Neither appears to be reported for sign-to-robot in what I found (absence of evidence from a limited search, not proof).

## 1. Sign / hand-gesture recognition to command robot arms

| Work | How it differs from SignVLA |
|---|---|
| SignVLA (2606.20857) and SignVLA (2602.22514), see above | Closest work. Same pipeline shape. Abstract-level: alphabet/command signs, one sign language, LLM/VLA text bridge. No language-agnostic concept layer, no false-activation or rejection evaluation stated in the abstract (**check full text**). |
| Lim, Sa, MacDonald, Ahn, *A Sign Language Recognition System with Pepper, Lightweight-Transformer, and LLM* (arXiv 2309.16898, 2023) | ASL to Pepper humanoid with co-speech gesture reply. HRI conversation, not manipulation, no shared concepts. |
| Muhtadin et al., *Hand Gesture Recognition for Collaborative Robots Using Lightweight Deep Learning in Real-Time Robotic Systems* (arXiv 2507.10055, ISITIA 2025) | 8 hand gestures, 1,103-parameter model, UR5. Lightweight and robot-commanded, but not sign language, no semantic layer. |
| *Research on Deep Learning-Based Human-Robot Static/Dynamic Gesture-Driven Control Framework* (PMC12693889) | Static+dynamic gesture control framework, no sign language. Authors/venue not verified **[UNVERIFIED]**. |
| Springer chapter *Simulation of Gesture Recognition Control Robotic Arm Based on Mediapipe and OpenCV* (doi 10.1007/978-981-96-7277-6_25) | MediaPipe gesture to simulated arm. Authors not verified **[UNVERIFIED]**. |
| MDPI Machines 13(3):182, *Gesture-Controlled Robotic Arm for Small Assembly Lines* (2025) | Gesture arm control for assembly. Authors not verified **[UNVERIFIED]**. |
| Waveshare RoArm-M2-Pro vendor demo (OpenCV+MediaPipe pan-tilt gesture control) | Vendor demo only. Shows RoArm plus MediaPipe is trivial. |
| Review: *Computer vision-based hand gesture recognition for human-robot interaction: a review*, Complex & Intelligent Systems (2023/24) | Survey. Cite for HRI gesture background. Authors not verified **[UNVERIFIED]**. |

Verdict: novelty of "gesture/sign controls a low-cost arm" is zero.

## 2. Sign / gesture conditioned VLA / LLM

| Work | Difference |
|---|---|
| Bai et al., SignVLA (2606.20857) | Above. Sign to text to VLA. |
| Tan et al., SignVLA gloss-free (2602.22514) | Above. |
| Guo et al., *GesVLA: Gesture-Aware VLA Model with Embedded Representations* (arXiv 2605.22812, May 2026) | Deictic/pointing gestures as a parallel modality inside the VLA latent space. Not sign language and not discrete commands. |
| Liu et al., *GIVE: Grounding Human Gestures in VLA Models* (arXiv 2606.13435, Jun 2026) | Pointing gestures overlaid on observations. Not sign language. |
| *DeicticVLA* (arXiv 2608.28108) **[UNVERIFIED: authors/abstract not fetched]** | Deictic gestures in one VLA. |
| OpenVLA (Kim et al., arXiv 2406.09246), RT-2 (Brohan et al., arXiv 2307.15818), pi0 (Black et al., arXiv 2410.24164) **[UNVERIFIED: IDs from memory, not fetched this session]** | Text-instruction VLAs. Sign would enter only as text. |
| LaMI (arXiv 2401.15174), TalkWithMachines (arXiv 2412.15462) | LLM multimodal HRI interfaces (speech/gesture). Only titles seen **[UNVERIFIED details]**. |

Verdict: "sign to text to VLA" is taken. Gesture-in-VLA is a crowded 2026 topic (pointing/deixis).

## 3. Language-agnostic / cross-sign-language semantics

| Work | Difference |
|---|---|
| Wei & Chen, *Improving Continuous SLR with Cross-Lingual Signs* (arXiv 2308.10809, ICCV 2023) | Shared vision encoder over multiple sign languages, cross-lingual sign mapping. Closest to the "different languages, same concept" idea, but for recognition accuracy, not a robot command interface. |
| Joshi et al., *CISLR* (EMNLP 2022, aclanthology 2022.emnlp-main.707) | ISL word corpus with prototype one-shot learner transferring from ASL. Direct precedent for ASL/ISL transfer. |
| Jiang et al., *SignCLIP* (arXiv 2407.01264, 2024) **[UNVERIFIED authors/venue]** | Contrastive sign-text embedding over 20+ sign languages. Direct precedent for language-agnostic sign embeddings. |
| *SONAR-SLT* (arXiv 2510.19398) **[UNVERIFIED authors]** | Multilingual language-agnostic sentence-embedding supervision for SLT. |
| Granero-Moya et al., *Zero-Shot Cross-Lingual Recognition of Sign Language Handshapes* (arXiv 2609.18772, EMNLP 2026 SLP workshop) | Shared phonological features ASL to LSC. Handshape-level, not commands. |
| Artiaga et al., *Cross-Sign Language Transfer Learning Using Domain Adaptation with Multi-scale Temporal Alignment* (arXiv 2608.16804; MTAP vol. 83, 2024 per page) | Transfer between sign languages. Venue/date inconsistency on the page; verify. |
| *Q-BridgeNet* (arXiv 2607.11215) **[UNVERIFIED]**; *Sign Language Recognition in the Age of LLMs* (arXiv 2604.11225) **[UNVERIFIED]** | Related multilingual/LLM work. |

Verdict: language-agnostic sign representation is not novel in itself. Note the difference: the literature learns shared *visual/embedding* spaces; your layer is a hand-specified *concept/command* vocabulary. Also caution: many ISL and ASL signs for "left/right/up/down/yes/no" are indexical or iconic and may already be near-identical. Your 14-class set may be *too easy* to show a real cross-language claim.

## 4. Lightweight landmark temporal classifiers, ISL datasets

| Work | Note |
|---|---|
| Srivastava et al., *INCLUDE: A Large Scale Dataset for ISL Recognition* (ACM MM 2020, doi 10.1145/3394171.3413528) | 263 words, 4,287 videos (per search summary). Authors **[UNVERIFIED]**. |
| CISLR (EMNLP 2022) | ~4,765 words, ~7,050 videos. Multiple signers/sources. |
| *ISL-CSLTR* (Elakkiya, Natarajan; Mendeley data kcmpdxky7p) | Sentence-level, 7 signers. Verified from dataset page/snippet only. |
| *Indian Sign Language Recognition Using Mediapipe Holistic* (arXiv 2304.10256) and *Comprehensive Approach to ISLR: LSTM and MediaPipe Holistic* (EAI Trans. AI & Robotics) | MediaPipe+LSTM ISL is already published. |
| *MP-GestLSTM* (Int. J. Systems Science: Operations & Logistics? tandfonline 10.1080/21642583.2025.2587853) **[UNVERIFIED venue]** | MediaPipe+LSTM gestures. |
| *Hierarchical Windowed Graph Attention Network and a Large Scale Dataset for Isolated ISLR* (arXiv 2407.14224) | Large isolated ISL dataset; I could not confirm it is what you call "ISL500". **"ISL500" is [UNVERIFIED]. Check your repo's source/licence.** |
| *TransSLR: A Lightweight Transformer for SLR* (arXiv 2608.06407) **[UNVERIFIED]**, *EfficientSign* (arXiv 2604.08694) **[UNVERIFIED]** | Lightweight recognisers. |
| *iSign* (arXiv 2407.05404), *ISLTranslate* (ACL Findings 2023) | ISL benchmarks. |

Verdict: a Bi-GRU on MediaPipe landmarks is a baseline, not a contribution. Note ISLRTC is a dictionary (government video lexicon), not a benchmark with a paper. Check its terms of use before redistributing clips.

## 5. Gesture control with sim-to-real and false-activation / rejection metrics

- Open-set gesture recognition exists but mostly for sEMG: Liu et al., *Towards Open-set Gesture Recognition via Feature Activation Enhancement and Orthogonal Prototype Learning* (arXiv 2312.02535). Modality differs; it gives the method vocabulary (open-set, prototypes).
- Search surfaced statements that HRI systems should reject transient/false gestures and use activation regions (activation-gesture / "wake" gating), but I did not pin down a specific vision-based sign-to-robot paper reporting false-activations per hour with sim-to-real comparison. **No such paper was found; this is the gap, not a proof of absence.**
- Precedent the reviewer will cite: wake-word/keyword-spotting evaluation (false accepts per hour), open-set recognition (AUROC, FPR@95TPR), teleoperation latency studies. Borrow these metrics explicitly.

## 6. Honest novelty assessment

**Not novel (be blunt):**
- MediaPipe + GRU/LSTM sign classifier. Done many times, including for ISL.
- Sign/gesture to robot arm command mapping, including on RoArm-class hardware.
- Sign to text to LLM/VLA to arm: taken by the two SignVLA papers.
- "First sign-driven VLA": false.
- Cross-sign-language shared embeddings: SignCLIP, Wei & Chen, CISLR.
- MuJoCo sim then a physical arm: standard practice, not a contribution alone.
- 14-class accuracy on your own recordings: not evidence of anything by itself.

**Potentially defensible contribution:** a *safety-and-interface study*: a language-agnostic command layer that decouples the sign recogniser from the robot, evaluated as a *safety-critical command channel* (false activation, unknown rejection, latency, controllability, sim-to-real) with signer-independent, cross-language evidence. The novelty is the measured protocol and the empirical results, not the components.

## 7. Strongest reframings (pick one)

1. **Safe sign-command gating**: an evaluation-first paper. Contribution = protocol + benchmark for false-activation rate (per minute of idle/free-signing/conversation footage), open-set rejection of out-of-vocabulary signs and non-sign motion, and activation policy (wake sign, temporal stability, confidence/entropy thresholds).
2. **Language-agnostic command interface with measured interchangeability**: show that ISL and ASL recognisers (or one joint model) drive the *same* robot layer, with a quantified concept-level accuracy and robot task success independent of source language. Needs at least two languages with real data.
3. **Sim-to-real gap for sign-commanded manipulation**: quantify latency budget per stage, command-success and trajectory error gap between MuJoCo and RoArm, under a constrained action set.
4. **Sign as a low-cost, non-verbal channel for constrained (non-VLA) robots vs VLA**: an ablation of "concept layer + scripted primitives" vs "sign to text to OpenVLA", on controllability, latency and safety. The honest likely result (VLA adds latency and unpredictability for a 14-command vocabulary) is itself a publishable, useful finding.
5. **Dataset/resource paper**: signer-independent, multi-language, robot-command vocabulary set (e.g. ISL+ASL parallel signs for the same 14 concepts, with idle/distractor negative class). Only viable if you record multiple signers.

Recommended: 1 + 2 combined, with 4 as an ablation. Avoid framing as "first" anything.

## 8. Minimum experiments for reviewer acceptance

1. **Signer-independent splits**: leave-one-signer-out (or fixed held-out signers), report mean and std across folds. Minimum about 8-10 signers; fewer is a pilot only.
2. **Session-level splits** even within a signer; no overlapping augmentations across train/test.
3. **Baselines**: MLP or kNN/DTW on landmarks, TCN, Transformer, and (if permitted) a pretrained ISLR model; report parameter count and CPU latency. Show Bi-GRU is justified or else drop the claim.
4. **False activation**: N hours of idle/free-motion/conversation/other-sign video, report false commands per hour with CIs, and the threshold trade-off curve (FA/hour vs command recall).
5. **Unknown rejection**: held-out signs (other ISL/ASL vocabulary) and non-sign gestures; AUROC and FPR at 95% TPR for softmax, energy, and entropy baselines.
6. **Latency**: end-to-end decomposition (camera, MediaPipe, window fill, classifier, gating, command, actuation), p50/p95 on the target hardware.
7. **Cross-language claim**: at least a second sign language on the same 14 concepts, from independent sources and signers; report per-language and concept-level accuracy and downstream task success, plus train-on-A/test-on-B transfer.
8. **Sim-to-real**: identical task suite in MuJoCo and on RoArm, N repetitions each (use 20+), success rate, task time, position error, with confidence intervals; log failure causes.
9. **Controllability**: human-subject or at least multi-operator trials (task completion, time, error, SUS/NASA-TLX if possible), including a deaf/HoH signer or an advisor from the community, and ethics approval where required.
10. **VLA ablation**: with and without OpenVLA in the loop, same commands. If the VLA is optional and unused in the headline results, do not claim it as a contribution.
11. **Direct comparison** against the two SignVLA papers' reported settings where possible.

## 9. Dataset advice for publication

- **Single signer is disqualifying for any generalisation claim.** A model trained and tested on one person's recordings measures memorisation of that person's hand, camera and lighting.
- **30-40 samples/class** is a pilot scale. With random per-sample splits, near-duplicate consecutive windows/clips from one session leak across train and test and inflate accuracy to ~100%. Split by signer, then by session/day, *before* any windowing or augmentation, and freeze split lists in the repo.
- **Sliding-window leakage**: several 30-frame windows cut from one video must all stay in the same split.
- **Source mixing**: your data mixes clips from CISLR, ISLRTC dictionary, ISL500 (**verify identity**), etc. Report per-source composition; dictionary clips are studio-quality with a single reference signer, so training on them and testing on webcam data is a domain shift you should measure, not hide.
- **Idle/negative class** and **hard negatives** (transition motions, resting hand, other signs) are mandatory for false-activation claims.
- **Ecological variation**: lighting, background, camera height, distance, handedness (left-handed signers), clothing, framing at 30 fps vs webcam jitter.
- Report class balance, confusion matrix (expect back/front, up/down, left/right, open/close pairs to confuse), and per-signer accuracy.
- **Licensing/ethics**: check CISLR/ISLRTC/ISL500 licences before redistributing; you committed videos to git (data/isl/...), and the git status shows they are staged for deletion, so decide deliberately. Consent forms for any new recordings. Involve Deaf community members and consider that a 14-gesture set is a command vocabulary, *not* "sign language recognition"; avoid claiming to translate ISL.
- **Terminology**: call the system "sign-based command interface" not "sign language translation", to avoid overclaiming.

## 10. References (URLs)

| # | Citation | URL | Status |
|---|---|---|---|
| 1 | Bai et al., SignVLA: Real-Time SL-Guided Robotic Manipulation via Attention LSTM and VLA, arXiv 2606.20857, 2026 | https://arxiv.org/abs/2606.20857 | verified (abstract) |
| 2 | Tan et al., SignVLA: A Gloss-Free VLA Framework..., arXiv 2602.22514, 2026 | https://arxiv.org/abs/2602.22514 | verified (abstract) |
| 3 | Lim et al., SLR System with Pepper, Lightweight-Transformer, and LLM, arXiv 2309.16898, 2023 | https://arxiv.org/abs/2309.16898 | verified |
| 4 | Muhtadin et al., Hand Gesture Recognition for Cobots, ISITIA 2025, arXiv 2507.10055 | https://arxiv.org/abs/2507.10055 | verified |
| 5 | Guo et al., GesVLA, arXiv 2605.22812, 2026 | https://arxiv.org/abs/2605.22812 | verified |
| 6 | Liu et al., GIVE, arXiv 2606.13435, 2026 | https://arxiv.org/abs/2606.13435 | verified |
| 7 | DeicticVLA, arXiv 2608.28108 | https://arxiv.org/pdf/2608.28108 | title only |
| 8 | Wei & Chen, Cross-Lingual Signs, ICCV 2023, arXiv 2308.10809 | https://arxiv.org/abs/2308.10809 | verified |
| 9 | Joshi et al., CISLR, EMNLP 2022 | https://aclanthology.org/2022.emnlp-main.707/ | verified (search) |
| 10 | INCLUDE, ACM MM 2020 | https://dl.acm.org/doi/10.1145/3394171.3413528 | verified (search); authors not checked |
| 11 | Granero-Moya et al., Zero-Shot Cross-Lingual Handshapes, arXiv 2609.18772 | https://arxiv.org/abs/2609.18772 | verified |
| 12 | Artiaga et al., Cross-Sign Language Transfer w/ Domain Adaptation, arXiv 2608.16804 | https://arxiv.org/abs/2608.16804 | verified; date oddity |
| 13 | SignCLIP, arXiv 2407.01264 | https://arxiv.org/abs/2407.01264 | title only |
| 14 | SONAR-SLT, arXiv 2510.19398 | https://arxiv.org/abs/2510.19398 | title only |
| 15 | Liu et al., Open-set Gesture Recognition (sEMG), arXiv 2312.02535 | https://arxiv.org/abs/2312.02535 | verified |
| 16 | ISL Mediapipe Holistic, arXiv 2304.10256 | https://arxiv.org/abs/2304.10256 | title only |
| 17 | Hierarchical Windowed Graph Attention + large ISL dataset, arXiv 2407.14224 | https://arxiv.org/abs/2407.14224 | title only |
| 18 | ISL-CSLTR (Mendeley) | https://data.mendeley.com/datasets/kcmpdxky7p/1 | dataset page seen |
| 19 | iSign, arXiv 2407.05404 | https://arxiv.org/abs/2407.05404 | title only |
| 20 | OpenVLA, arXiv 2406.09246; RT-2, arXiv 2307.15818; pi0, arXiv 2410.24164 | https://arxiv.org/abs/2406.09246 etc. | IDs from memory, UNVERIFIED |
| 21 | Gesture-Driven Control Framework, PMC12693889 | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12693889/ | title only |
| 22 | Gesture-Controlled Robotic Arm for Small Assembly Lines, Machines 13(3):182 | https://www.mdpi.com/2075-1702/13/3/182 | title only |
| 23 | HRI hand gesture review, Complex & Intelligent Systems | https://link.springer.com/article/10.1007/s40747-023-01173-6 | title only |
| 24 | Waveshare RoArm-M2-Pro | https://iotcart.in/product/waveshare-roarm-m2-pro-desktop-4-dof-robotic-arm-kit-based-on-esp32 | product page |

Items with "title only" were seen in search results; details not fetched. Do not cite them with authors/venues until you open them.
