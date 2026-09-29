# experiments/ — old scripts kept for reference, NOT part of the tested pipeline

- `openarm/` — OpenArm bimanual MuJoCo demos (need the `openarm_mujoco` package, path `.venv/share/...`).
- `roarm_prototypes/` — early one-off RoArm sim/keyboard/gesture scripts. Run them from the repo root.
- `hardware_untested/` — serial controller for the physical RoArm. **Do not run on the real arm**: it imports the deleted
  `src/` package and its JSON command IDs (T=102 for xyz, T=103 for gripper) could not be verified against the
  Waveshare docs. Rebuild it on top of `signvla.robot.roarm_sim.PRIMITIVES` at the hardware stage.
