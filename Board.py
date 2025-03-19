from abc import ABC, abstractmethod

from BoardState import BoardState
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

    def 