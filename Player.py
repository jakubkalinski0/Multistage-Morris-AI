from enum import Enum

class Player(Enum):
    """Player representation."""
    NONE = 0
    WHITE = 1
    BLACK = 2
    def opponent(self, Player):
        """Returns the opponent of the player."""
        if self == Player.WHITE:
            return Player.BLACK
        elif self == Player.BLACK:
            return Player.WHITE
        return Player.NONE