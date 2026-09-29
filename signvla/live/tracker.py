"""Live camera loop: MediaPipe -> 30-frame window -> encoder -> gates -> debounce -> concept -> UDP."""
import socket
import time
from collections import deque

import cv2
import numpy as np

from signvla import config
from signvla.encoder.model import load_checkpoint
from signvla.live.recognizer import Debouncer, Recognizer
from signvla.perception.features import extract_features
from signvla.perception.hands import draw_landmarks, get_hands_detector
from signvla.semantics.concepts import resolve


def run(camera=config.CAMERA_INDEX, language=config.SIGN_LANGUAGE, send=True, log=None):
    """`log`: optional path; appends one CSV row per fired command (for latency/false-activation studies)."""
    model, gestures = load_checkpoint()
    rec, deb = Recognizer(model, gestures), Debouncer()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if send else None

    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera (macOS: grant camera permission to your terminal).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    detector = get_hands_detector()
    window = deque(maxlen=config.SEQUENCE_LENGTH)
    shown, last_reason, sent = "-", "warming up", "-"
    print("Live. Press Q in the camera window to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)  # same mirroring as recording
        t0 = time.time()
        results = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        window.append(extract_features(results))
        draw_landmarks(frame, results)

        if len(window) == config.SEQUENCE_LENGTH:
            pred = rec.predict(np.asarray(window))
            shown = f"{pred.label} {pred.confidence * 100:.0f}%" if pred.accepted else "-"
            last_reason = pred.reason
            fired = deb.update(pred, t0)
            if fired:
                concept = resolve(language, fired)
                latency_ms = (time.time() - t0) * 1000
                print(f"[FIRE] {fired} -> {concept.value if concept else 'NO CONCEPT'}  ({latency_ms:.0f} ms inference)")
                if concept:
                    sent = concept.value
                    if sock:
                        sock.sendto(concept.value.encode(), (config.UDP_HOST, config.UDP_PORT))
                    if log:
                        with open(log, "a") as f:
                            f.write(f"{time.time():.3f},{fired},{concept.value},{pred.confidence:.3f},{latency_ms:.1f}\n")

        cv2.putText(frame, f"sign: {shown}   ({last_reason})", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 128), 2)
        cv2.putText(frame, f"last command: {sent}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
        cv2.imshow("SignVLA live (Q = quit)", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    if sock:
        sock.sendto(b"QUIT", (config.UDP_HOST, config.UDP_PORT))
        sock.close()
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
