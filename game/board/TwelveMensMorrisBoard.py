from game.board.Board import Board

from game.board.Maps import TWELVE_MEN_GRAPH,TWELVE_MEN_MILLS



class TwelveMensMorrisBoard(Board):
    BOARD_SIZE = len(TWELVE_MEN_GRAPH)
    PIECES_PER_PLAYER = 12

    def __init__(self):
        graph=TWELVE_MEN_GRAPH
        mills=TWELVE_MEN_MILLS
        super().__init__(board_graph=graph, pieces_per_player=self.PIECES_PER_PLAYER, board_mills=mills)

    def print_map_2d(self):
        print(r"0---------1---------2")
        print(r"| \       |       / |")
        print(r"|   3-----4-----5   |")
        print(r"|   | \   |   / |   |")
        print(r"|   |  6--7--8  |   |")
        print(r"|   |  |     |  |   |")
        print(r"9--10-11     12-13-14")
        print(r"|   |  |     |  |   |")
        print(r"|   |  15-16-17 |   |")
        print(r"|   | /   |   \ |   |")
        print(r"|   18----19----20  |")
        print(r"| /       |       \ |")
        print(r"21--------22--------23")
