import copy

from GamePhase import GamePhase
from Player import Player


class BoardState:
    def __init__(self, board_size: int, pieces_per_player: int):
        self.board_size = board_size
        self.board = [Player.NONE] * board_size
        self.pieces_per_player = pieces_per_player
        self.game_phase_for_player = {
            Player.WHITE: GamePhase.PLACEMENT,
            Player.BLACK: GamePhase.PLACEMENT
        }
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

    def is_game_over(self) -> bool:
        """Check if the game is over based on pieces count."""
        white_done_placing = self.pieces_left_to_place_by_player[Player.WHITE] == 0
        black_done_placing = self.pieces_left_to_place_by_player[Player.BLACK] == 0

        if white_done_placing and black_done_placing:
            return (self.pieces_from_player_currently_on_board[Player.WHITE] < 3 or
                    self.pieces_from_player_currently_on_board[Player.BLACK] < 3)
        return False

    def __str__(self):
        """Return a string representation of the board state."""
        pieces_repr = {
            Player.NONE: '.',
            Player.WHITE: 'W',
            Player.BLACK: 'B'
        }

        board_str = "Board:\n"
        for i in range(self.board_size):
            board_str += f"{i}: {pieces_repr[self.board[i]]}  "
            if (i + 1) % 8 == 0 or i == self.board_size - 1:
                board_str += "\n"

        board_str += f"\nCurrent player: {self.current_player.name}\n"
        board_str += f"White phase: {self.game_phase_for_player[Player.WHITE].name}\n"
        board_str += f"Black phase: {self.game_phase_for_player[Player.BLACK].name}\n"
        board_str += f"White pieces to place: {self.pieces_left_to_place_by_player[Player.WHITE]}\n"
        board_str += f"Black pieces to place: {self.pieces_left_to_place_by_player[Player.BLACK]}\n"
        board_str += f"White pieces on board: {self.pieces_from_player_currently_on_board[Player.WHITE]}\n"
        board_str += f"Black pieces on board: {self.pieces_from_player_currently_on_board[Player.BLACK]}\n"
        board_str += f"Need to remove piece: {self.need_to_remove_piece}\n"

        return board_str