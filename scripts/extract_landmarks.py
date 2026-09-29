"""Re-extract landmarks from data/raw_videos/<gesture>/*.mp4 with the canonical feature code.
  python -m scripts.extract_landmarks
Only the 6 classes recorded by the old recorder have videos (hello, home, no, pick, place, yes).
Output overwrites data/landmarks/<gesture>/<sample>.npy - run scripts.fix_landmarks first to keep a backup.
"""
import cv2
import numpy as np

from signvla import config
from signvla.perception.features import extract_features
from signvla.perception.hands import get_hands_detector


def resample(seq, n):
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == 0:
        return np.zeros((n, config.FEATURES_PER_FRAME), dtype=np.float32)
    idx = np.linspace(0, len(seq) - 1, n)
    return np.stack([np.interp(idx, np.arange(len(seq)), seq[:, d]) for d in range(seq.shape[1])], 1).astype(np.float32)


def main():
    videos = sorted(config.RAW_VIDEOS_DIR.glob("*/*.mp4"))
    if not videos:
        raise SystemExit("No videos in data/raw_videos/")
    detector = get_hands_detector()
    for i, v in enumerate(videos, 1):
        cap, feats = cv2.VideoCapture(str(v)), []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            feats.append(extract_features(detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        cap.release()
        out = config.LANDMARKS_DIR / v.parent.name / f"{v.stem}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, resample(feats, config.SEQUENCE_LENGTH))
        print(f"[{i}/{len(videos)}] {v.parent.name}/{v.stem}")
    detector.close()


if __name__ == "__main__":
    main()
