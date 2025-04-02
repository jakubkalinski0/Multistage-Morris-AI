import copy

from GamePhase import GamePhase
from Player import Player
from Position import Position


class BoardState:
    def __init__(self, board_size: int, pieces_per_player: int):
        self.board_size = board_size
        self.board = [Player.NONE]*board_size
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

    def __str__(self):
        return NotImplemented