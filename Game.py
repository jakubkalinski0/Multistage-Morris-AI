from Board import Board
from BoardState import BoardState
from NineMensMorrisBoard import NineMensMorrisBoard
from ThreeMensMorrisBoard import ThreeMensMorrisBoard
from SixMensMorrisBoard import SixMensMorrisBoard
from TwelveMensMorrisBoard import TwelveMensMorrisBoard


class Game:
    def __init__(self):
        self.board = self._choose_board()
        # self.state = BoardState(self.board)

    def _choose_board(self):
        standard_available_boards = {1:ThreeMensMorrisBoard, 2:SixMensMorrisBoard, 3:NineMensMorrisBoard, 4:TwelveMensMorrisBoard}
        chosen_board = int(input(f"You can choose one of the 4 following boards by typing in their index (1, 2, 3, 4):\n"
                             f"    [1] Three Men's Morris,\n"
                             f"    [2] Six Men's Morris,\n"
                             f"    [3] Nine Men's Morris,\n"
                             f"    [4] Twelve Men's Morris.\n").strip().lower())
        return standard_available_boards.get(chosen_board)

    def _setup_players(self):
        return NotImplemented # Na razie mamy jedynie ludzkich graczy więc nie ma sensu implementowanie tego

    def start(self):
        while not self.state.is_game_over():
            self.play_turn()
            self.state.current_player = self.state.current_player.switch_player(self.players)
        self.end_game()

    def play_turn(self):
        if
        return NotImplemented

    def end_game(self):
        return NotImplemented