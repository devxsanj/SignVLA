"""Record labelled gesture samples (landmarks + video) in the canonical feature space.
  python -m scripts.record --gestures pick place --samples 40
Keys: SPACE record | N next gesture | R delete last | Q quit
Then:  python -m scripts.build_dataset
"""
import argparse
import time

import cv2
import numpy as np

from signvla import config
from signvla.perception.features import extract_features
from signvla.perception.hands import draw_landmarks, get_hands_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gestures", nargs="+", default=config.GESTURES)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--countdown", type=int, default=3)
    a = ap.parse_args()

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    detector = get_hands_detector()
    gi, state, t_start = 0, "READY", 0.0
    feats, frames = [], []

    def paths(g, n):
        return (config.LANDMARKS_DIR / g / f"sample_{n:03d}.npy", config.RAW_VIDEOS_DIR / g / f"sample_{n:03d}.mp4")

    def count(g):
        d = config.LANDMARKS_DIR / g
        return len(list(d.glob("sample_*.npy"))) if d.exists() else 0

    while cap.isOpened():
        g = a.gestures[gi]
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # MUST match live inference
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        f = extract_features(results)
        draw_landmarks(frame, results)
        now = time.time()

        if state == "COUNTDOWN":
            left = a.countdown - int(now - t_start)
            if left <= 0:
                state, feats, frames = "RECORDING", [], []
            else:
                msg = f"get ready {left}"
        if state == "RECORDING":
            feats.append(f)
            frames.append(frame.copy())
            msg = f"REC {len(feats)}/{config.SEQUENCE_LENGTH}"
            if len(feats) == config.SEQUENCE_LENGTH:
                n = count(g) + 1
                npy, mp4 = paths(g, n)
                npy.parent.mkdir(parents=True, exist_ok=True)
                mp4.parent.mkdir(parents=True, exist_ok=True)
                np.save(npy, np.asarray(feats, dtype=np.float32))
                h, w = frames[0].shape[:2]
                out = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), config.FPS, (w, h))
                for fr in frames:
                    out.write(fr)
                out.release()
                print(f"saved {g} #{n}")
                state = "READY"
        if state == "READY":
            msg = "SPACE = record"
        cv2.putText(frame, f"{g.upper()}  {count(g)}/{a.samples}   {msg}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("SignVLA recorder", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" ") and state == "READY":
            state, t_start = "COUNTDOWN", now
        if k == ord("n") and state == "READY":
            gi = (gi + 1) % len(a.gestures)
        if k == ord("r") and state == "READY" and count(g):
            for p in paths(g, count(g)):
                p.unlink(missing_ok=True)
    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
