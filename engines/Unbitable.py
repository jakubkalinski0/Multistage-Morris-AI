import networkx as nx
import random
from typing import Optional
from game.board.Board import Board
from game.board.BoardState import BoardState
from game.util.Move import Move

class GraphAgent:
    """
    Agent wybierający losowy ruch prowadzący do stanu wygranego dla aktualnego gracza na podstawie grafu.
    """

    def __init__(self, board: Board, graph_path: str):
        self.board = board
        self.graph = nx.read_adjlist(graph_path, create_using=nx.DiGraph())

    def get_best_move(self, state: BoardState) -> Optional[Move]:
        legal_moves = self.board.get_legal_moves(state)
        if not legal_moves:
            return None

        current_player = state.current_player.value  # 1 lub 2
        winning_mod = 1 if current_player == 1 else 2

        winning_moves = []
        for move in legal_moves:
            new_state = self.board.make_move(state, move)
            next_node = new_state.to_int()
            
            if str(next_node + winning_mod) in self.graph.nodes:
                winning_moves.append(move)
        if winning_moves:
            return random.choice(winning_moves)
        else:
            return random.choice(legal_moves)