import networkx as nx
from game.board.Board import Board
from game.board.BoardState import BoardState
from game.util.Player import Player
from game.board.SixMensMorrisBoard import SixMensMorrisBoard
import sys
sys.setrecursionlimit(10000)


class GameGraphGenerator:
    def __init__(self, board: Board):
        self.board = board
        self.graph = nx.DiGraph()
        state = board.get_initial_board_state()
        self.graph.add_node(state.to_int())
        self.graph.nodes[state.to_int()][1]=False
        state_int = state.to_int()

    def generate_graph(self, state: BoardState, state_int: int = None):
        if self.board.get_winner(state) is not Player.NONE:
            print(len(self.graph.nodes), "nodes")
            return  
        legal_moves = self.board.get_legal_moves(state)
        for move in legal_moves:
            next_state = self.board.make_move(state, move)
            next_state_int = next_state.to_int()
            
            if self.graph.has_node(next_state_int) == False:
                self.graph.add_edge(state_int, next_state_int)
                self.generate_graph(next_state, next_state_int)
            else:
                self.graph.add_edge(state_int, next_state_int)

        
    def save_graph_to_file(self, filename: str):
        """Save the graph to a text file using NetworkX's built-in method."""
        nx.write_adjlist(self.graph, filename)


if __name__ == "__main__":

    #board = ThreeMensMorrisBoard()
    board = SixMensMorrisBoard()
    generator = GameGraphGenerator(board)

    initial_state = board.get_initial_board_state()
    generator.generate_graph(initial_state, initial_state.to_int())
    # Save the graph to a file
    generator.save_graph_to_file("Sixgame_graph.txt")
    print("Game graph generated and saved to Sixgame_graph.txt")

