"""Record a continuous landmark STREAM (not windows) for streaming evaluation.
Saves data/streams/<name>.npz with features (T, 126) and timestamps (T,).
  python -m scripts.record_stream --name idle_day1_a --minutes 10
Idle streams: behave naturally and never sign. Command streams: sign normally (log what you signed).
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
    ap.add_argument("--name", required=True)
    ap.add_argument("--minutes", type=float, default=10)
    a = ap.parse_args()
    out = config.DATA_DIR / "streams" / f"{a.name}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    det = get_hands_detector()
    feats, ts, t0 = [], [], time.time()
    while time.time() - t0 < a.minutes * 60:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        res = det.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        feats.append(extract_features(res))
        ts.append(time.time() - t0)
        draw_landmarks(frame, res)
        cv2.putText(frame, f"{a.name}  {int(ts[-1])}/{int(a.minutes * 60)}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.imshow("SignVLA stream recorder (Q = stop)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release(); cv2.destroyAllWindows(); det.close()
    np.savez_compressed(out, features=np.asarray(feats, np.float32), timestamps=np.asarray(ts, np.float64))
    print(f"saved {out}: {len(feats)} frames, {ts[-1] / 60:.1f} min")


if __name__ == "__main__":
    main()
