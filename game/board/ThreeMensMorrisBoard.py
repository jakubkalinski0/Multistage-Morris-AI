from game.board.Board import Board
from game.util.Player import Player
from game.board.Maps import THREE_MEN_GRAPH,THREE_MEN_MILLS


class ThreeMensMorrisBoard(Board):
    BOARD_SIZE = len(THREE_MEN_GRAPH)
    PIECES_PER_PLAYER = 3

    def __init__(self):
        graph=THREE_MEN_GRAPH
        mills=THREE_MEN_MILLS
        super().__init__(board_graph=graph, pieces_per_player=self.PIECES_PER_PLAYER, board_mills=mills)


    def get_piece_symbol(self, player):
        if player == Player.WHITE:
            return "W"
        elif player == Player.BLACK:
            return "B"
        else:
            return "·"
        
    def print_map_2d(self):
        symbols = [self.get_piece_symbol(self.board_state.board[i]) for i in range(9)]

        print("\nBoard with pieces:")
        print(f"{symbols[0]}---{symbols[1]}---{symbols[2]}")
        print("| \\ | / |")
        print(f"{symbols[3]}---{symbols[4]}---{symbols[5]}")
        print("| / | \\ |")
        print(f"{symbols[6]}---{symbols[7]}---{symbols[8]}")
        print("Legend: W = White, B = Black, · = Empty")
