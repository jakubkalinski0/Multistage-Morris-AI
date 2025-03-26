from Board import Board

from Maps import THREE_MEN_GRAPH,THREE_MEN_MILLS


class SixMensMorrisBoard(Board):  
    BOARD_SIZE = len(THREE_MEN_GRAPH)
    PIECES_PER_PLAYER = 3

    def __init__(self):
        graph=THREE_MEN_GRAPH
        mills=THREE_MEN_MILLS
        super().__init__(board_graph=graph, pieces_per_player=self.PIECES_PER_PLAYER, board_mills=mills)


    def print_map_2d(self):
        print("0---1---2")
        print("| \ | / |") 
        print("3---4---5")
        print("| / | \ |")
        print("6---7---8")
