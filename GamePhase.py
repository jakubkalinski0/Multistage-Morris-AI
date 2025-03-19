from enum import Enum

class GamePhase(Enum):
    """Game phase representation."""
    PLACEMENT = 0
    MOVEMENT = 1
    FLYING = 2