"""MuJoCo RoArm viewer that executes semantic commands from run_live.
  Terminal 1:  mjpython -m scripts.run_sim          (macOS needs mjpython for the viewer)
  Terminal 2:  python -m scripts.run_live
  Headless self-test of every primitive:  python -m scripts.run_sim --headless
"""
import argparse

from signvla.robot import roarm_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()
    if a.headless:
        model = roarm_sim.load_model()
        for name in roarm_sim.PRIMITIVES:
            r = roarm_sim.run_headless(name, model)
            print(f"{name:<14} success={r['success']}  sim_s={r['sim_seconds']:.2f}  final_err={r['final_error']:.3f} rad")
    else:
        roarm_sim.run_viewer()


if __name__ == "__main__":
    main()
