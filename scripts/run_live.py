"""Live recognition -> semantic command over UDP.
  python -m scripts.run_live                # send commands to the simulator
  python -m scripts.run_live --no-send      # just watch the recognizer (debugging)
  python -m scripts.run_live --log runs/session1.csv
"""
import argparse

from signvla import config
from signvla.live.tracker import run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--language", default=config.SIGN_LANGUAGE)
    ap.add_argument("--no-send", action="store_true")
    ap.add_argument("--log")
    a = ap.parse_args()
    run(a.camera, a.language, send=not a.no_send, log=a.log)


if __name__ == "__main__":
    main()
