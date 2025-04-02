import copy

from GamePhase import GamePhase
from Player import Player
from Position import Position


class BoardState:
    def __init__(self, board_size: int, pieces_per_player: int):
        self.board_size = board_size
        self.who_wins = 0
        self.board = [Player.NONE]*board_size
        self.pieces_per_player = pieces_per_player
        self.game_phase = GamePhase.PLACEMENT # white starts first placment change when black ends phase

        self.pieces_left_to_place_by_player = {
            Player.WHITE: pieces_per_player,
            Player.BLACK: pieces_per_player
        }
        self.pieces_from_player_currently_on_board = {
            Player.WHITE: 0,
            Player.BLACK: 0
        }
        self.current_player = Player.WHITE
        self.need_to_remove_piece = False
        self.move_history = []

    def get_current_phase_for_player(self, player: Player) -> GamePhase:
        return self.game_phase_for_player[player]

    def get_player_at_position(self, position: int) -> Player:
        return self.board[position]

    def is_position_empty(self, position: int) -> bool:
        return self.board[position] == Player.NONE

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return NotImplemented

    def to_int(self):
        """Converts the board state to an integer representation."""
        ## The integer representation is constructed as follows:
        ## - The first digit represents who wins (0 for none, 1 for white, 2 for black).
        ## - The second digit represents the current player (0 for none, 1 for white, 2 for black).
        ## - The next digits represent the state of each position on the board (0 for none, 1 for white, 2 for black).
        ## - The next 3 digits represents the number of pieces left to place for each player (0-3**3).
        ## - The last 3 digits represents the number of pieces left to place for each player (0-3**3).
        result = self.who_wins
        result+= 3 * self.current_player.value
        for i in range(self.board_size):
            result+= self.board[i].value * (3 ** (i+2))
        result += 3 ** (self.board_size + 2) * self.pieces_left_to_place_by_player[Player.WHITE]
        result += 3 ** (self.board_size + 5) * self.pieces_left_to_place_by_player[Player.BLACK]
        return result
    
    def from_int(self, int_representation):
        """Converts an integer representation back to the board state."""
        self.who_wins = int_representation % 3
        int_representation //= 3
        self.current_player = Player(int_representation % 3)
        int_representation //= 3
        for i in range(self.board_size):
            self.board[i] = Player(int_representation % 3)
            int_representation //= 3
        self.pieces_left_to_place_by_player[Player.WHITE] = int_representation % 3**3
        int_representation //= 3**3
        self.pieces_left_to_place_by_player[Player.BLACK] = int_representation % 3**3

