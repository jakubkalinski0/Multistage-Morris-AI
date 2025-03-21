from abc import ABC, abstractmethod
from typing import List

from BoardState import BoardState
from Move import Move
from Player import Player
from Position import Position


class Board(ABC):
    def __init__(self, board_size: int, pieces_per_player: int):
        self.board_size = board_size
        self.pieces_per_player = pieces_per_player
        self.positions = [Position(i) for i in range(board_size)]
        self.mills = []

        self._initialize_connections()
        self._initialize_mills()

    @abstractmethod
    def _initialize_connections(self):
        pass

    @abstractmethod
    def _initialize_mills(self):
        pass

    def get_initial_board_state(self) -> BoardState:
        return BoardState(self.board_size, self.pieces_per_player)

    def check_if_move_creates_mill(self, state: BoardState, position: Position, player: Player) -> bool:
        return NotImplemented

    def get_legal_moves(self, state: BoardState) -> List[Position]:
        return NotImplemented

    def make_move(self, state: BoardState, move: Move, player: Player) -> BoardState:
        return NotImplemented

    def check_if_game_is_over(self, state: BoardState) -> bool:
        return NotImplemented

    def get_winner(self, state: BoardState) -> Player:
        return NotImplemented