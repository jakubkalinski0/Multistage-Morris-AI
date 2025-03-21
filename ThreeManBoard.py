from Board import Board


class SixMensMorrisBoard(Board):  

    def __init__(self):
        super().__init__(board_size=9, pieces_per_player=3)