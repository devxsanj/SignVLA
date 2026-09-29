"""Language-agnostic semantic layer:  sign (language, label) -> Concept -> robot primitive.

The robot side only ever receives a `Concept`; it never learns which sign language
or which sign label produced it. Supporting a new sign language means adding one
lexicon table here - nothing downstream changes.
"""
from enum import Enum


class Concept(str, Enum):
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_FORWARD = "move_forward"
    MOVE_BACKWARD = "move_backward"
    GRIPPER_OPEN = "gripper_open"
    GRIPPER_CLOSE = "gripper_close"
    PICK = "pick"
    PLACE = "place"
    GO_HOME = "go_home"
    GREET = "greet"
    AFFIRM = "affirm"
    NEGATE = "negate"


# sign-language -> {sign label -> Concept}. Add "asl", "bsl", ... here.
LEXICONS = {
    "isl": {
        "left": Concept.MOVE_LEFT, "right": Concept.MOVE_RIGHT,
        "up": Concept.MOVE_UP, "down": Concept.MOVE_DOWN,
        "front": Concept.MOVE_FORWARD, "back": Concept.MOVE_BACKWARD,
        "open": Concept.GRIPPER_OPEN, "close": Concept.GRIPPER_CLOSE,
        "pick": Concept.PICK, "place": Concept.PLACE, "home": Concept.GO_HOME,
        "hello": Concept.GREET, "yes": Concept.AFFIRM, "no": Concept.NEGATE,
    },
    "asl": {},  # TODO(multi-sign phase): fill once an ASL command set is recorded
}

# Concept -> named robot primitive (see signvla/robot/roarm_sim.py TRAJECTORIES).
CONCEPT_TO_PRIMITIVE = {
    Concept.MOVE_LEFT: "move_left", Concept.MOVE_RIGHT: "move_right",
    Concept.MOVE_UP: "move_up", Concept.MOVE_DOWN: "move_down",
    Concept.MOVE_FORWARD: "move_front", Concept.MOVE_BACKWARD: "move_back",
    Concept.GRIPPER_OPEN: "open_gripper", Concept.GRIPPER_CLOSE: "close_gripper",
    Concept.PICK: "pick_object", Concept.PLACE: "place_object",
    Concept.GO_HOME: "home", Concept.GREET: "wave",
    Concept.AFFIRM: "nod_head", Concept.NEGATE: "shake_head",
}


def resolve(language, label):
    """Sign label -> Concept, or None if the sign is not in that language's lexicon."""
    return LEXICONS.get(language, {}).get(label)


def primitive_for(concept):
    return CONCEPT_TO_PRIMITIVE[Concept(concept)]
