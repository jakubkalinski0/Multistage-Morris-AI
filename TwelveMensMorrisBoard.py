from Board import Board

from Maps import TWELVE_MEN_GRAPH,TWELVE_MEN_MILLS



class TwelveMensMorrisBoard(Board):
    BOARD_SIZE = len(TWELVE_MEN_GRAPH)
    PIECES_PER_PLAYER = 12

    def __init__(self):
        graph=TWELVE_MEN_GRAPH
        mills=TWELVE_MEN_MILLS
        super().__init__(board_graph=graph, pieces_per_player=self.PIECES_PER_PLAYER, board_mills=mills)

    def print_map_2d(self):
        print("0---------1---------2")
        print("| \       |       / |")
        print("|   3-----4-----5   |")
        print("|   | \   |   / |   |")
        print("|   |  6--7--8  |   |")
        print("|   |  |     |  |   |")
        print("9--10-11     12-13-14")
        print("|   |  |     |  |   |")
        print("|   |  15-16-17 |   |")
        print("|   | /   |   \ |   |")
        print("|   18----19----20  |")
        print("| /       |       \ |")
        print("21--------22--------23")
