from typing import Optional

from MoveType import MoveType
from Position import Position

class Move:
    """Representation of movement in the game."""
    def __init__(self, move_type: MoveType, from_position: Optional[Position] = None, to_position: Optional[Position] = None, remove_checker_from_position: Optional[Position] = None):
        self.move_type = move_type
        self.from_position = from_position
        self.to_position = to_position
        self.remove_checker_from_position = remove_checker_from_position

    def __eq__(self, other):
        if not isinstance(other, Move):
            return False
        return (self.move_type == other.move_type and
                self.from_position == other.from_position and
                self.to_position == other.to_position and
                self.remove_checker_from_position == other.remove_checker_from_position)

    def __hash__(self):
        return hash((self.move_type, self.from_position, self.to_position, self.remove_checker_from_position))