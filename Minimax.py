from typing import Tuple, Optional, List
import time
import math
from Board import Board
from BoardState import BoardState
from Player import Player
from Move import Move
from GamePhase import GamePhase


class Minimax:
    """Implementation of Minimax algorithm with Alpha-Beta pruning for the Morris game."""

    def __init__(self, board: Board, max_depth: int = 4):
        self.board = board
        self.max_depth = max_depth
        self.nodes_evaluated = 0
        self.start_time = 0

    def get_best_move(self, state: BoardState, max_time_seconds: float = 5.0) -> Optional[Move]:
        """Find the best move for the current player using Minimax with Alpha-Beta pruning."""
        self.nodes_evaluated = 0
        self.start_time = time.time()

        player = state.current_player
        legal_moves = self.board.get_legal_moves(state)

        if not legal_moves:
            return None

        best_move = None
        best_value = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in legal_moves:
            new_state = self.board.make_move(state, move)
            value = self.minimax(new_state, 1, alpha, beta, False, player, max_time_seconds)

            if value > best_value:
                best_value = value
                best_move = move

            alpha = max(alpha, best_value)

            # Check if time limit exceeded
            if time.time() - self.start_time > max_time_seconds:
                print(f"Time limit reached. Evaluated {self.nodes_evaluated} nodes.")
                break

        print(f"Minimax completed. Best move value: {best_value}. Evaluated {self.nodes_evaluated} nodes.")
        return best_move

    def minimax(self, state: BoardState, depth: int, alpha: float, beta: float,
                is_maximizing: bool, original_player: Player, max_time_seconds: float) -> float:
        """Minimax implementation with Alpha-Beta pruning."""
        self.nodes_evaluated += 1

        # Check termination conditions
        if time.time() - self.start_time > max_time_seconds:
            # Time limit reached
            return self.evaluate_state(state, original_player,depth)

        if depth >= self.max_depth or self.board.check_if_game_is_over(state):
            return self.evaluate_state(state, original_player,depth)

        legal_moves = self.board.get_legal_moves(state)
        if not legal_moves:
            # No legal moves available, evaluate current state
            return self.evaluate_state(state, original_player,depth)

        if is_maximizing:
            value = float('-inf')
            for move in legal_moves:
                new_state = self.board.make_move(state, move)
                # If removal move created, continue maximizing
                next_is_maximizing = not new_state.need_to_remove_piece if new_state.current_player != state.current_player else True
                value = max(value, self.minimax(new_state, depth + 1, alpha, beta, next_is_maximizing, original_player,
                                                max_time_seconds))
                alpha = max(alpha, value)
                # if alpha >= beta:
                #     break  # Beta cut-off
            return value
        else:
            value = float('inf')
            for move in legal_moves:
                new_state = self.board.make_move(state, move)
                # If removal move created, continue minimizing
                next_is_maximizing = new_state.need_to_remove_piece if new_state.current_player != state.current_player else False
                value = min(value, self.minimax(new_state, depth + 1, alpha, beta, next_is_maximizing, original_player,
                                                max_time_seconds))
                beta = min(beta, value)
                # if beta <= alpha:
                #     break  # Alpha cut-off
            return value

    def evaluate_state(self, state: BoardState, player: Player, deph:int) -> float:
        """
        Evaluate the board state from the perspective of the given player.
        Returns a higher value for better positions for the player.
        """
        # Game over check
        if self.board.check_if_game_is_over(state):
            winner = self.board.get_winner(state)
            if winner == player:
                return 1000 + (self.max_depth-deph)*100  # Win
            elif winner == player.opponent():
                return -1000 - (self.max_depth-deph)*100  # Loss
            else:
                return 0  # Draw

        # Determine game phase for evaluation
        phase = state.get_current_phase_for_player(player)
        opponent = player.opponent()

        # Basic piece count difference (always important)
        piece_diff = state.pieces_from_player_currently_on_board[player] - state.pieces_from_player_currently_on_board[
            opponent]

        # Mill count
        player_mills = self.count_mills(state, player)
        opponent_mills = self.count_mills(state, opponent)
        mill_diff = player_mills - opponent_mills

        # Potential mills (pieces that are one move away from forming a mill)
        player_potential_mills = self.count_potential_mills(state, player)
        opponent_potential_mills = self.count_potential_mills(state, opponent)
        potential_mill_diff = player_potential_mills - opponent_potential_mills

        # Movement freedom (number of empty positions adjacent to player's pieces)
        player_mobility = self.count_mobility(state, player)
        opponent_mobility = self.count_mobility(state, opponent)
        mobility_diff = player_mobility - opponent_mobility

        # Different weights based on game phase
        if phase == GamePhase.PLACEMENT:
            # In placement phase, prioritize strategic positions and potential mills
            return (
                    3 * piece_diff +
                    10 * mill_diff +
                    2 * potential_mill_diff +
                    self.evaluate_position_control(state, player)
            )
        elif phase == GamePhase.MOVEMENT:
            # In movement phase, mobility becomes more important
            return (
                    5 * piece_diff +
                    10 * mill_diff +
                    2 * potential_mill_diff +
                    2 * mobility_diff
            )
        else:  # FLYING phase
            # In flying phase, aggressively prioritize reducing opponent's pieces
            return (
                    8 * piece_diff +
                    6 * mill_diff +
                    potential_mill_diff
            )

    def count_mills(self, state: BoardState, player: Player) -> int:
        """Count the number of mills the player has formed."""
        mill_count = 0
        counted_positions = set()

        for mill in self.board.mills:
            # Check if all positions in the mill belong to the player
            if (state.get_player_at_position(mill[0]) == player and
                    state.get_player_at_position(mill[1]) == player and
                    state.get_player_at_position(mill[2]) == player):

                # Only count each position once even if it belongs to multiple mills
                for pos in mill:
                    if pos not in counted_positions:
                        counted_positions.add(pos)

                mill_count += 1

        return mill_count

    def count_potential_mills(self, state: BoardState, player: Player) -> int:
        """Count the number of potential mills (two pieces in a line with an empty third)."""
        potential_mills = 0

        for mill in self.board.mills:
            player_count = 0
            empty_count = 0

            for pos in mill:
                if state.get_player_at_position(pos) == player:
                    player_count += 1
                elif state.is_position_empty(pos):
                    empty_count += 1

            # Two player pieces and one empty position
            if player_count == 2 and empty_count == 1:
                potential_mills += 1

        return potential_mills

    def count_mobility(self, state: BoardState, player: Player) -> int:
        """Count the number of possible moves for the player's pieces."""
        mobility = 0

        for i in range(state.board_size):
            if state.get_player_at_position(i) == player:
                # Check each neighboring position
                for neighbor in self.board.graph.iterNeighbors(i):
                    if state.is_position_empty(neighbor):
                        mobility += 1

        return mobility

    def evaluate_position_control(self, state: BoardState, player: Player) -> float:
        """
        Evaluate the strategic value of positions controlled by the player.
        Some positions are more valuable as they belong to more potential mills.
        """
        value = 0

        # Count how many potential mills each position can form
        position_value = [0] * state.board_size
        for mill in self.board.mills:
            for pos in mill:
                position_value[pos] += 1

        # Add value for each controlled position
        for i in range(state.board_size):
            if state.get_player_at_position(i) == player:
                value += position_value[i]
            elif state.get_player_at_position(i) == player.opponent():
                value -= position_value[i]

        return value