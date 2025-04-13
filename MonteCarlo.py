import time
import random
from typing import Optional
from Board import Board
from BoardState import BoardState
from Player import Player
from Move import Move
from MorrisHeuristic import MorrisHeuristics


class MonteCarloTreeSearch:
    """Monte Carlo Tree Search for Morris game with time and depth constraints."""

    def __init__(self, board: Board, max_depth: int = 10):
        self.board = board
        self.max_depth = max_depth
        self.start_time = 0
        self.simulations = 0
        self.heuristic = MorrisHeuristics(board)

    def get_best_move(self, state: BoardState, max_time_seconds: float = 5.0) -> Optional[Move]:
        """Select the best move using Monte Carlo simulations within a time limit."""
        self.start_time = time.time()
        self.simulations = 0

        legal_moves = self.board.get_legal_moves(state)
        if not legal_moves:
            return None

        move_scores = {move: 0 for move in legal_moves}
        move_simulations = {move: 0 for move in legal_moves}

        while time.time() - self.start_time < max_time_seconds:
            move = random.choice(legal_moves)
            new_state = self.board.make_move(state, move)
            result = self.simulate_random_game(new_state, depth=0, original_player=state.current_player)
            move_scores[move] += result
            move_simulations[move] += 1
            self.simulations += 1

        # Pick move with best average score
        best_move = max(
            legal_moves,
            key=lambda m: move_scores[m] / move_simulations[m] if move_simulations[m] > 0 else -float('inf')
        )

        print(f"MCTS completed. Simulations: {self.simulations}")
        print(f"Best move: {best_move}, Score: {move_scores[best_move]}, Simulations: {move_simulations[best_move]}")
        return best_move

    def simulate_random_game(self, state: BoardState, depth: int, original_player: Player) -> float:
        """Simulate a random game until the end or maximum depth, return +1 for win, -1 for loss, 0 for draw."""
        current_state = state
        current_depth = depth

        while not self.board.check_if_game_is_over(current_state) and current_depth < self.max_depth:
            legal_moves = self.board.get_legal_moves(current_state)
            if not legal_moves:
                break

            move = random.choice(legal_moves)
            current_state = self.board.make_move(current_state, move)
            current_depth += 1

        winner = self.board.get_winner(current_state)
        if winner == original_player:
            return 10**10
        elif winner == original_player.opponent():
            return -(10**10)
        return self.heuristic.evaluate_state(current_state, original_player)
