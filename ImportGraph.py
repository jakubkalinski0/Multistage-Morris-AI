import networkx as nx
from ThreeMensMorrisBoard import ThreeMensMorrisBoard
from GameGraph import GameGraphGenerator
from Board import Board
from BoardState import BoardState
if __name__ == "__main__":


    graph=nx.read_adjlist("Threegame_graph.txt")
    print(graph.nodes["12963"])
    boaed=ThreeMensMorrisBoard()
    boaed.board_state.from_int(12891)
    boaed.print_map_2d()