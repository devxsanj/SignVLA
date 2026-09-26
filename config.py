"""
ISL Gesture Detection + RoArm-M2-Pro Configuration
"""

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw_videos"
LANDMARKS_DIR = DATA_DIR / "landmarks"

MODELS_DIR = PROJECT_ROOT / "models"
SIM_DIR = PROJECT_ROOT / "simulation"
ROARM_OFFICIAL_DIR = PROJECT_ROOT / "roarm_official"

ROARM_MODEL_XML = ROARM_OFFICIAL_DIR / "roarm_m2_pro.xml"

if not ROARM_MODEL_XML.exists():
    ROARM_MODEL_XML = SIM_DIR / "roarm_m2_pro.xml"

if not ROARM_MODEL_XML.exists():
    ROARM_MODEL_XML = SIM_DIR / "roarm_m2.xml"


RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
LANDMARKS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
SIM_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GESTURE VOCABULARY
# ============================================================

GESTURES = [
    "back",
    "close",
    "down",
    "front",
    "hello",
    "home",
    "left",
    "no",
    "open",
    "pick",
    "place",
    "right",
    "up",
    "yes",
]


# ============================================================
# GESTURE -> ROBOT ACTION
# ============================================================

GESTURE_ACTION_MAP = {

    # Basic directional gestures
    "left": "move_left",
    "right": "move_right",

    "up": "move_up",
    "down": "move_down",

    "front": "move_front",
    "back": "move_back",

    # End effector
    "open": "open_gripper",
    "close": "close_gripper",

    # Task
    "pick": "pick_object",
    "place": "place_object",

    # Demonstration gestures
    "hello": "wave",
    "yes": "nod_head",
    "no": "shake_head",

    # HOME = return to 90-degree rest
    # It does NOT move to sleep.
    "home": "home",
}


# ============================================================
# CAMERA / RECORDING
# ============================================================

SEQUENCE_LENGTH = 30
SAMPLES_PER_GESTURE = 30

COUNTDOWN_SECONDS = 3

CAMERA_INDEX = 0

FPS = 30

FRAME_WIDTH = 640
FRAME_HEIGHT = 480


# ============================================================
# MEDIAPIPE
# ============================================================

MAX_NUM_HANDS = 2

MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5


# ============================================================
# LANDMARK FEATURES
# ============================================================

NUM_HANDS = 2
LANDMARKS_PER_HAND = 21
COORDS_PER_LANDMARK = 3

TOTAL_FEATURES_PER_FRAME = (
    NUM_HANDS
    * LANDMARKS_PER_HAND
    * COORDS_PER_LANDMARK
)