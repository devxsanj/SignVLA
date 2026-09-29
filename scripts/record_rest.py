"""Record the 'rest' class: idle / transition footage that must NOT trigger the robot.
Streams the camera for --seconds and slices it into overlapping 30-frame windows
(saved as data/landmarks/rest/sample_XXX.npy, in time order).

While it runs, behave like a normal idle operator: hands resting on the desk or lap,
hands leaving/entering the frame, scratching, adjusting the camera, reaching for a cup,
starting/ending a sign WITHOUT completing it, talking with hands. Do NOT hold a real sign.
  python -m scripts.record_rest --seconds 90
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
    ap.add_argument("--seconds", type=int, default=90)
    ap.add_argument("--stride", type=int, default=15, help="frames between window starts")
    a = ap.parse_args()

    out_dir = config.LANDMARKS_DIR / config.REST_LABEL
    out_dir.mkdir(parents=True, exist_ok=True)
    start_n = len(list(out_dir.glob("sample_*.npy")))
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    detector = get_hands_detector()
    feats, t0 = [], time.time()
    while time.time() - t0 < a.seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        feats.append(extract_features(results))
        draw_landmarks(frame, results)
        cv2.putText(frame, f"REST recording {int(time.time() - t0)}/{a.seconds}s - act idle, no real signs",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.imshow("SignVLA rest recorder (Q = stop early)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    L = config.SEQUENCE_LENGTH
    feats = np.asarray(feats, dtype=np.float32)
    n = 0
    for s in range(0, len(feats) - L + 1, a.stride):
        n += 1
        np.save(out_dir / f"sample_{start_n + n:03d}.npy", feats[s:s + L])
    hand = (np.abs(feats).sum(1) > 0).mean() if len(feats) else 0
    print(f"{len(feats)} frames -> {n} rest windows saved ({hand * 100:.0f}% of frames had a hand)")


if __name__ == "__main__":
    main()
