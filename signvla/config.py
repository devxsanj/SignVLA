"""Project-wide constants. Everything that more than one module needs lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
LANDMARKS_DIR = DATA_DIR / "landmarks"
DATASET_PATH = DATA_DIR / "dataset.npz"
LABEL_MAP_PATH = DATA_DIR / "label_map.json"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "isl_gesture_model.pth"
REPORTS_DIR = ROOT / "reports"
ROARM_XML = ROOT / "assets" / "roarm_m2_pro" / "roarm_m2_pro.xml"

# ---- Sign vocabulary (ISL command set). Order = label index order. ----
GESTURES = [
    "back", "close", "down", "front", "hello", "home", "left",
    "no", "open", "pick", "place", "right", "up", "yes",
]
SIGN_LANGUAGE = "isl"

# ---- Sequence / feature shape ----
SEQUENCE_LENGTH = 30          # frames per sample (~1 s at 30 fps)
NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
FEATURES_PER_HAND = LANDMARKS_PER_HAND * 3        # 63
FEATURES_PER_FRAME = NUM_HANDS * FEATURES_PER_HAND  # 126

# Bump when the feature definition changes; old checkpoints are then refused.
FEATURE_VERSION = 2

# ---- Camera / MediaPipe ----
CAMERA_INDEX = 0
FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

# ---- Live gating (tuned in reports/, see docs/EVALUATION.md) ----
CONFIDENCE_THRESHOLD = 0.75   # min top-1 softmax probability
MARGIN_THRESHOLD = 0.30       # min (top-1 minus top-2) probability
MIN_HAND_FRAMES = 0.6         # fraction of the window that must contain a hand
STABLE_FRAMES = 3             # consecutive agreeing predictions before firing
COOLDOWN_SECONDS = 1.2        # min gap between two robot commands

# ---- Robot bridge (recognizer process -> simulator process) ----
UDP_HOST = "127.0.0.1"
UDP_PORT = 9876
