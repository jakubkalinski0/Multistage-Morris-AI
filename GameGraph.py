import networkx as nx
from typing import Dict, List
from Board import Board
from BoardState import BoardState
from Move import Move
from Player import Player
from SixMensMorrisBoard import SixMensMorrisBoard


class GameGraphGenerator:
    def __init__(self, board: Board):
        self.board = board
        self.graph = nx.DiGraph()

    def generate_graph(self, state: BoardState):
        """Recursively generate the game graph."""
        state_int = state.to_int()
        if state_int in self.graph:
            return

        self.graph.add_node(state_int)

        legal_moves = self.board.get_legal_moves(state)
        if not legal_moves:

            winner = self.board.get_winner(state)
            if winner == Player.WHITE:
                state_int += 1
            elif winner == Player.BLACK:
                state_int += 2
            self.graph.add_node(state_int)
            return

        for move in legal_moves:
            next_state = self.board.make_move(state, move)
            next_state_int = next_state.to_int()

            self.graph.add_edge(state_int, next_state_int, move=move)

            self.generate_graph(next_state)

    def evaluate_graph_with_minimax(self):
        """Evaluate the graph using the Minimax algorithm."""
        def minimax(state_int: int, is_maximizing: bool) -> int:
            evaluation = state_int % 3
            if evaluation != 0: 
                return evaluation

            if is_maximizing:
                value = -float('inf')
                for _, child in self.graph.out_edges(state_int):
                    value = max(value, minimax(child, False))
                if value == 1:
                    state_int += 1  # Mark as White win
                elif value == 2:
                    state_int += 2  # Mark as Black win
                return value
            else:
                value = float('inf')
                for _, child in self.graph.out_edges(state_int):
                    value = min(value, minimax(child, True))
                if value == 1:
                    state_int += 1  # Mark as White win
                elif value == 2:
                    state_int += 2  # Mark as Black win
                return value

        initial_state_int = self.board.get_initial_board_state().to_int()
        minimax(initial_state_int, True)

    def save_graph_to_file(self, filename: str):
        """Save the graph to a text file using NetworkX's built-in method."""
        nx.write_adjlist(self.graph, filename)


if __name__ == "__main__":
    # Initialize the board and generator
    board = SixMensMorrisBoard()
    generator = GameGraphGenerator(board)

    # Generate the graph
    initial_state = board.get_initial_board_state()
    generator.generate_graph(initial_state)

    # Evaluate the graph using Minimax
    generator.evaluate_graph_with_minimax()

    # Save the graph to a file
    generator.save_graph_to_file("Sixgame_graph.txt")