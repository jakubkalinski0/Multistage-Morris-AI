from enum import Enum

class MoveType(Enum):
    """Move representation."""
    PLACE = 0
    MOVE = 1
    REMOVE = 2