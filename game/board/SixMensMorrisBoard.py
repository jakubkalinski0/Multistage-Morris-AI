from game.board.Board import Board
import plotly.io as pio
import networkx as nx
import matplotlib.pyplot as plt

from game.board.Maps import SIX_MEN_GRAPH,SIX_MEN_MILLS

class SixMensMorrisBoard(Board):  
    BOARD_SIZE = len(SIX_MEN_GRAPH)
    PIECES_PER_PLAYER = 6

    def __init__(self):
        graph=SIX_MEN_GRAPH
        mills=SIX_MEN_MILLS
        super().__init__(board_graph=graph, pieces_per_player=self.PIECES_PER_PLAYER, board_mills=mills)


    def print_map_2d(self):
        print("0-------1-------2")
        print("|       |       |")
        print("|   3---4---5   |")
        print("|   |       |   |")
        print("6---7       8---9")
        print("|   |       |   |")
        print("|   10--11--12  |")
        print("|       |       |")
        print("13------14------15")

# map = SixMensMorrisBoard()
# map.print_map_2d()
# print(map.board_state.to_int())
# print(map.board_state.board)
# print(map.board_state.pieces_left_to_place_by_player)