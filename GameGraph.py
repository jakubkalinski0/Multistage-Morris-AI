import networkx as nx
from typing import Dict, List
from Board import Board
from BoardState import BoardState
from Move import Move
from Player import Player
from ThreeMensMorrisBoard import ThreeMensMorrisBoard
import sys
sys.setrecursionlimit(10000)

class GameGraphGenerator:
    def __init__(self, board: Board):
        self.board = board
        self.graph = nx.DiGraph()

    def generate_graph(self, state: BoardState, state_int: int = None):
        """Recursively generate the game graph."""
        # if len(self.graph.nodes) > 500:
        #     return
        if state_int is None:
            self.graph.add_node(state.to_int())
            state_int = state.to_int()

        if self.board.get_winner(state) is not Player.NONE:
            print(len(self.graph.nodes), "nodes")

            winner = self.board.get_winner(state)
            self.graph.nodes[state_int]['winner'] = winner
            
            return  
        legal_moves = self.board.get_legal_moves(state)
        for move in legal_moves:
            next_state = self.board.make_move(state, move)
            next_state_int = next_state.to_int()
            if self.graph.has_edge(state_int, next_state_int):
                continue
            self.graph.add_edge(state_int, next_state_int)
            self.generate_graph(next_state,next_state_int)
        
    def save_graph_to_file(self, filename: str):
        """Save the graph to a text file using NetworkX's built-in method."""
        nx.write_adjlist(self.graph, filename)


if __name__ == "__main__":
    # Initialize the board and generator
    board = ThreeMensMorrisBoard()
    generator = GameGraphGenerator(board)

    # Generate the graph
    initial_state = board.get_initial_board_state()
    generator.generate_graph(initial_state)

    # Evaluate the graph using Minimax
    #generator.evaluate_graph_with_minimax()

    # Save the graph to a file
    generator.save_graph_to_file("Threegame_graph.txt")
    print("Game graph generated and saved to Threegame_graph.txt")

