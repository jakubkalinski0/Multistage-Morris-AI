from enum import Enum


class Player(Enum):
    """Player representation."""
    NONE = 0
    WHITE = 1
    BLACK = 2

    def opponent(self):
        """Returns the opponent of the player."""
        if self == Player.WHITE:
            return Player.BLACK
        elif self == Player.BLACK:
            return Player.WHITE
        return Player.NONE

    def switch_player(self, players):
        """Switches between two players."""
        if self == players[0]:
            return players[1]
        return players[0]