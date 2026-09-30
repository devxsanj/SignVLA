# Novelty angles: prior-art check (2026-09-30)

Method: WebSearch (US-only index) plus arXiv abstract-page fetches. "V" = abstract page fetched by me;
"U" = seen only in a search snippet (UNVERIFIED; do not cite until you open it). Absence of hits is
evidence, not proof: a ~20-query sweep cannot rule out a paper indexed under different words.
Re-run the searches in the last section the week you submit.

Already known (only differentiated here): SignVLA 2602.22514 and 2606.20857 (sign->text->VLA; the
2606.20857 abstract mentions a temporal stabilizer but no rejection class, calibration, or false-activation
metric), GesVLA 2605.22812 (gesture as extra VLA modality for target grounding; abstract has no rejection
or calibration), GIVE 2606.13435.

## A. Risk-controlled sign command interface (conformal / selective / open-set, false activations per hour)

Closest prior art:
- RCIP, Lidard et al., 2403.15959 (V): set-valued intent prediction with conformal-style risk calibration for
  HRI; decides act vs. ask. Not sign, not command false-activation rate, not per-hour.
- On-Demand Myoelectric Control with Wake Gestures, 2402.10050 (V): >99.9% rejection of non-target input during
  daily activities; a wake gesture, no statistical bound. Closest in *metric spirit* (idle false activations).
- Underwater Diver-Robot Interaction, 2609.23392 (V abstract): "intertrial false trigger rate" 35% -> 7.7% with
  hold-based authorisation (from a search snippet, U for the numbers). Heuristic, no guarantee.
- ASL Trigger Recognition in Mixed Activity/Signing Sequences (RF), 2111.05480 (V): sign "wake words" vs.
  non-signing activity, false alarms vs. sign stroke count. Closest sign-specific idle-rejection work; RF
  sensor, no robot, no bound.
- Conformal/selective abstention for VLAs: BOKBO 2605.30660 (V), ReconVLA 2604.16677 (V), Calibrated Action
  Abstention (ResearchGate, no arXiv id found; U). These abstain on *policy actions*, not on the human's
  command channel.
- Wake-word literature reports FAR per hour (industry standard, e.g. <0.5/h): the metric exists, just not for sign.

Verdict: **PARTLY TAKEN.** Conformal risk control for robots, and idle false-trigger rates for gestures/EMG, both
exist. I found no paper that (1) gives a calibrated bound on false activations for a *sign/gesture command
channel* and (2) reports it as false commands per hour over long idle streams. The gap is the combination and
the sign-language setting, not conformal prediction itself. Caveat: a distribution-free bound holds per
window under exchangeability; per-hour claims need a temporal argument (dependence between consecutive
frames) or an empirical, held-out idle-hours estimate with a Clopper-Pearson upper limit. Say that plainly.

## B. Concept gate between sign recogniser and VLA

Closest: SignVLA x2 (sign->text->VLA, no vetting); INSIGHT 2510.01389 (V: token-level uncertainty to
decide when a VLA asks for help; gates on the VLA's internals, not the instruction source); SAFECAST
2608.04246 (V: failure detection for VLA policies, contrast sets on language and vision); BOKBO, ReconVLA (above);
"Uncertainty-Calibrated Safety Gating for VLA" (PMC13210885, U, page blocked by captcha).
Verdict: **PARTLY TAKEN.** VLA-side abstention is crowded. Gating the *upstream* human-input encoder and
handing the VLA only vetted, template instructions, with a three-way ablation (sign->text->VLA vs.
gate->VLA vs. direct primitives), I did not find. Weakness: it is an engineering pattern; novelty is the
ablation and the measured false-activation cost of the un-gated baseline. Also note that with a 14-command
closed vocabulary, "vetted instruction" = a template lookup; a reviewer will say the VLA adds nothing.
You need a VLA task where instruction content matters (object choice) or drop VLA as a headline.

## C. Cross-sign-language interchangeability at the robot-outcome level

Closest: Zero-Shot Cross-Lingual Sign Language Handshape Recognition, 2609.18772 (V: ASL->LSC via phonological
features, recognition-level); SONAR-SLT (WMT25, U: language-agnostic semantic vectors supervise sign
translation); "Sign Language: Towards Sign Understanding for Robot Autonomy" 2506.02556 (U: reads
navigational *signage*, not sign language; do not confuse).
Verdict: **NOVEL at the robot-outcome level, but expensive.** I found no paper evaluating ISL and ASL signs
producing identical robot behaviour. Risks: you need a second signer/dataset (ASL) with matching concepts;
14 commands rarely have clean ISL/ASL equivalents; with one signer this is a data-collection project. Feasible
only as a small pilot (3-5 concepts, public ASL clips such as WLASL-style data, or a second signer who knows ASL). Do not claim
"language-agnostic" from ISL alone.

## D. Sim-to-real gap for sign-commanded low-cost arms with controllability metrics

Found only generic low-cost-arm sim2real (SO-101 sim2real repo, LiteArm-MuJoCo, U) and MediaPipe gesture-arm
demos (U). No sign-commanded sim-vs-real comparison found.
Verdict: **NOVEL but weak.** A 4-DOF arm executing 15 primitives has a small, uninteresting sim-to-real gap
(mostly latency and joint-limit clipping). Fold into B/A as a "same interface, sim and real" table, not a
contribution on its own.

## E. Methodology pitfalls audit (feature mismatch, single-signer random-split leakage)

Leakage is well known: signer-independent vs. random split gaps are reported (e.g. 96.4% vs. 85.2%;
99.7% vs. 68.2% on KArSL-100; both from a snippet, U, source papers not opened). Training/serving skew for
landmark pipelines: no dedicated paper found.
Verdict: **TAKEN as a claim, usable as evidence.** Do not sell it as a contribution; report it as a
measured, honest section (your own numbers: what accuracy the leak inflates, and what the feature fix
recovers). That is a strength for credibility, and it is cheap because you already have the repo history.

## F. Other angles

1. **Idle-time benchmark for sign commands** ("hours of non-command hands"): release a small dataset of
   rest/chatter/reach/scratch/eating signer behaviour with false-trigger-per-hour protocol, plus tooling. I
   found nothing equivalent for sign (2111.05480 is RF, 3 activities). Feasible with one person recording
   2-3 hours; low cost, high reuse. Strongest add-on to A.
2. **Command-level latency-vs-safety trade-off curves** (hold time, confirm sign, wake sign): 2609.23392 and
   2402.10050 cover parts for other modalities; a sign-specific "confirm sign" protocol study is open.
3. Deaf/hard-of-hearing user study: not feasible unaided; do not claim accessibility outcomes without it.

## Candidate titles (string searches)

1. "Silent Until Certain: Risk-Controlled Sign Commands for a Low-Cost Robot Arm" - exact-string search: no match.
2. "Sign, Then Gate: A Calibrated Concept Layer Between Sign Recognition and Robot Action" - exact-string search: no match.
3. "False Commands per Hour: Auditing Sign-Language Robot Interfaces During Idle Time" - I did not search this
   exact string; the near-variant "false commands per hour sign language robot" returned only SignVLA and
   unrelated hits, none using the phrase. Re-check before use.
Also probed with no hits: "Concept Gate sign language robot", "Sign to Concept to Action", "Say Nothing Until Sure".
Caveat: web search does not index arXiv titles exhaustively; also check arXiv listing search and Google Scholar.
Recommended: title 3 (metric-first, most defensible) or 1 (memorable).

## Recommendation: ONE framing

**"Idle-safe sign commands": A + F1 as the core, B as the system context, E as a credibility section.
Drop C to a 'future work / pilot' sentence, drop D as a claim.**

Claim (keep it modest): for a closed-vocabulary sign command interface, a calibrated abstaining classifier
plus a rest class gives a measurable, bounded false-activation rate over long idle streams, at a stated cost in
missed commands and latency; and gating the downstream executor (primitive or VLA) on that vetted output
removes the false robot motions that a sign->text->VLA baseline incurs.

Minimum experiments (3-4 weeks, one signer, one arm):
1. Data (week 1): 2-3 h idle/chatter stream (talking with hands, reaching, phone, keyboard), split by
   *session/day*, never by frame. Plus command clips in 3+ sessions. Hold out whole sessions for calibration
   and for test.
2. Baselines (week 1-2): argmax; max-softmax threshold; temperature-scaled threshold; conformal (split, threshold
   chosen on calibration sessions for target false-accept alpha); + temporal persistence (k consecutive windows).
   Report false activations/h (with Clopper-Pearson or bootstrap-over-sessions CI), missed-command rate,
   command latency, on the held-out sessions.
3. Coverage check (week 2): does the nominal bound hold on a *different day/lighting/clothing*? Show it
   fail or hold honestly; a documented failure under shift is publishable as an audit.
4. Executor ablation (week 2-3): sim (MuJoCo) with a scripted injected stream: (a) direct primitives,
   (b) sign->text->fixed template, (c) gate->executor, optionally (d) gate->OpenVLA if it runs. Metrics: unintended
   motions/h, task success on commanded trials, end-to-end latency. Real arm: a small confirmation run
   (e.g. 20 commands + 30 min idle) to show the sim result is not an artefact.
5. Audit section (week 3): random-split vs. session-split accuracy and train/inference feature-mismatch numbers you already have.

Optional, only if time: 3-5 concept pilot with a second signer (ASL) for C.

## What a workshop reviewer will attack

- One signer, one camera, one arm: no generalisation claim. Frame as a protocol and audit, not a system that "works for ISL users".
- Conformal guarantees need exchangeability; frames within a stream are dependent and idle behaviour shifts
  across days. "Bounded false activations per hour" must be an empirical, session-held-out estimate with CIs,
  not a theorem. Do not use "guarantee" unqualified.
- 14 commands is a toy vocabulary; the rest class is user-defined, so open-set behaviour is only as good
  as your idle recordings. Include a genuinely unseen idle session.
- Without deaf/HoH users, no accessibility claim. Signer status (were you fluent?) must be disclosed; 
  "ISL" signs performed by a non-native signer are a validity threat.
- Gate vs. VLA: with a closed vocabulary, a template lookup makes the VLA redundant; the gate only "matters" if
  the VLA task has content beyond the command label.
- Novelty against RCIP/BOKBO/wake-gesture work: pre-empt by citing them and stating the delta (sign channel,
  per-hour idle metric, executor ablation).
- Sim-to-real: 4-DOF, small gap; reported only as a sanity check.

## References

| Id / URL | Short title | Status |
|---|---|---|
| arXiv 2602.22514 | SignVLA (gloss-free) | V (given) |
| arXiv 2606.20857 | SignVLA (attention LSTM) | V |
| arXiv 2605.22812 | GesVLA | V |
| arXiv 2606.13435 | GIVE | given, not re-checked |
| arXiv 2403.15959 | RCIP, risk-calibrated HRI intent | V |
| arXiv 2402.10050 | Myoelectric wake gestures, false activations | V |
| arXiv 2609.23392 | Underwater diver-robot gestures, false trigger | V (abstract); numbers U |
| arXiv 2111.05480 | ASL trigger recognition, RF sensing | V |
| arXiv 2609.01662 | PACT, typed action admission in HRC | V (not sign, not conformal; loosely related) |
| arXiv 2605.30660 | BOKBO, conformal abstention for VLA | V |
| arXiv 2604.16677 | ReconVLA, conformal VLA | V |
| arXiv 2510.01389 | INSIGHT, VLA help triggers | V |
| arXiv 2608.04246 | SAFECAST, VLA failure detection | V |
| arXiv 2609.18772 | Zero-shot cross-lingual handshapes ASL->LSC | V |
| ResearchGate 404216409 | Calibrated Action Abstention (VLA) | U (no arXiv id seen) |
| PMC13210885 | Uncertainty-Calibrated Safety Gating for VLA | U (page blocked) |
| WMT25 2025.wmt-1.18 | SONAR-SLT | U |
| arXiv 2506.02556 | Sign understanding for robot autonomy (signage) | U, not sign language |
| arXiv 2608.06252 | Bangla SLR, signer/leak discussion | U |
| arXiv 2509.03690 | Low-cost ASL robotic hand (robot *signs*) | U, irrelevant direction |
| arXiv 2505.24266 | SignBot (humanoid signs) | U, irrelevant direction |
| arXiv 1901.09192 | SelectiveNet | U (well known; verify before citing) |
| arXiv 2107.11277 | ML with reject option survey | U |
| arXiv 2512.12844 | Selective conformal risk control | U |
| github BruceZhang111/sim2real; nexform-tech/litearm-mujoco | low-cost arm sim2real | U |
| The 96.4/85.2 and 99.7/68.2 leakage numbers | signer-independent gaps | U (source papers not opened) |
