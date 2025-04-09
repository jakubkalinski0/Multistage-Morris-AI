from typing import List, Dict, Set
from Board import Board
from BoardState import BoardState
from Player import Player
from GamePhase import GamePhase


class MorrisHeuristics:
    """Heuristic for evaluating Morris game states."""

    def __init__(self, board: Board):
        self.board = board
        # Initialize position importance based on the number of lines it belongs to
        self.position_importance = self._calculate_position_importance()

    def _calculate_position_importance(self) -> List[int]:
        """Calculate importance of each position based on number of potential mills it forms."""
        importance = [0] * self.board.board_size
        for mill in self.board.mills:
            for pos in mill:
                importance[pos] += 1
        return importance

    def evaluate_opening_phase(self, state: BoardState, player: Player) -> float:
        """
        Evaluation for the opening (placement) phase.
        Focus on controlling strategic positions and setting up mills.
        """
        opponent = player.opponent()
        score = 0

        # 1. Number of pieces on the board
        player_pieces = state.pieces_from_player_currently_on_board[player]
        opponent_pieces = state.pieces_from_player_currently_on_board[opponent]
        score += (player_pieces - opponent_pieces) * 5

        # 2. Formed mills
        player_mills = self._count_mills(state, player)
        opponent_mills = self._count_mills(state, opponent)
        score += (player_mills - opponent_mills) * 20

        # 3. Potential mills (two pieces in a line with the third position empty)
        player_potential = self._count_potential_mills(state, player)
        opponent_potential = self._count_potential_mills(state, opponent)
        score += (player_potential - opponent_potential) * 10

        # 4. Control of strategic positions
        for i in range(state.board_size):
            if state.board[i] == player:
                score += self.position_importance[i] * 3
            elif state.board[i] == opponent:
                score -= self.position_importance[i] * 3

        # 5. Blocked opponent moves
        opponent_blocked = self._count_blocked_positions(state, opponent)
        score += opponent_blocked * 2

        return score

    def evaluate_midgame_phase(self, state: BoardState, player: Player) -> float:
        """
        Evaluation for the midgame (movement) phase.
        Focus on mobility, blocking opponent, and creating mills.
        """
        opponent = player.opponent()
        score = 0

        # 1. Number of pieces on the board
        player_pieces = state.pieces_from_player_currently_on_board[player]
        opponent_pieces = state.pieces_from_player_currently_on_board[opponent]
        score += (player_pieces - opponent_pieces) * 10

        # 2. Formed mills
        player_mills = self._count_mills(state, player)
        opponent_mills = self._count_mills(state, opponent)
        score += (player_mills - opponent_mills) * 20

        # 3. Potential mills
        player_potential = self._count_potential_mills(state, player)
        opponent_potential = self._count_potential_mills(state, opponent)
        score += (player_potential - opponent_potential) * 5

        # 4. Mobility (number of moves available)
        player_mobility = self._count_mobility(state, player)
        opponent_mobility = self._count_mobility(state, opponent)
        score += (player_mobility - opponent_mobility) * 2

        # 5. Blocked opponent pieces
        opponent_blocked = self._count_blocked_positions(state, opponent)
        score += opponent_blocked * 3

        # 6. Double mills (configurations where moving one piece back and forth creates mills)
        player_double_mills = self._count_double_mill_configurations(state, player)
        score += player_double_mills * 15

        return score

    def evaluate_endgame_phase(self, state: BoardState, player: Player) -> float:
        """
        Evaluation for the endgame (flying) phase.
        Focus on reducing opponent pieces and maintaining mobility.
        """
        opponent = player.opponent()
        score = 0

        # 1. Number of pieces (very important in endgame)
        player_pieces = state.pieces_from_player_currently_on_board[player]
        opponent_pieces = state.pieces_from_player_currently_on_board[opponent]
        score += (player_pieces - opponent_pieces) * 25

        # 2. If opponent has exactly 3 pieces, prioritize blocking them
        if opponent_pieces == 3:
            opponent_blocked = self._count_blocked_positions(state, opponent)
            score += opponent_blocked * 15

        # 3. Mills and potential mills
        player_mills = self._count_mills(state, player)
        opponent_mills = self._count_mills(state, opponent)
        score += (player_mills - opponent_mills) * 15

        player_potential = self._count_potential_mills(state, player)
        opponent_potential = self._count_potential_mills(state, opponent)
        score += (player_potential - opponent_potential) * 10

        # 4. Mobility in endgame is less important for player if they can fly
        # but critical to restrict opponent
        opponent_mobility = self._count_mobility(state, opponent)
        score -= opponent_mobility * 5

        # 5. Winning configuration detection
        if player_pieces == 3 and opponent_pieces == 3:
            # In 3 vs 3 scenario, analyze winning configurations
            winning_position = self._evaluate_winning_position(state, player)
            score += winning_position * 30

        return score

    def evaluate_state(self, state: BoardState, player: Player) -> float:
        """
        Main evaluation function that selects appropriate phase-specific evaluation.
        """
        # First check if game is over
        if self.board.check_if_game_is_over(state):
            winner = self.board.get_winner(state)
            if winner == player:
                return 1  # Win
            elif winner == player.opponent():
                return -1  # Loss
            return 0  # Draw

        # Determine the current game phase for evaluation
        phase = state.get_current_phase_for_player(player)

        if phase == GamePhase.PLACEMENT:
            return self.evaluate_opening_phase(state, player)
        elif phase == GamePhase.MOVEMENT:
            return self.evaluate_midgame_phase(state, player)
        else:  # FLYING phase
            return self.evaluate_endgame_phase(state, player)

    def _count_mills(self, state: BoardState, player: Player) -> int:
        """Count the number of mills formed by the player."""
        mill_count = 0
        for mill in self.board.mills:
            if (state.board[mill[0]] == player and
                    state.board[mill[1]] == player and
                    state.board[mill[2]] == player):
                mill_count += 1
        return mill_count

    def _count_potential_mills(self, state: BoardState, player: Player) -> int:
        """Count potential mills (two pieces in a line with an empty spot)."""
        potential_count = 0
        for mill in self.board.mills:
            player_pieces = 0
            empty_positions = 0
            for pos in mill:
                if state.board[pos] == player:
                    player_pieces += 1
                elif state.board[pos] == Player.NONE:
                    empty_positions += 1

            if player_pieces == 2 and empty_positions == 1:
                potential_count += 1
        return potential_count

    def _count_mobility(self, state: BoardState, player: Player) -> int:
        """Count the number of moves available to the player."""
        mobility = 0
        for i in range(state.board_size):
            if state.board[i] == player:
                # Check each neighboring position
                for neighbor in self.board.graph.iterNeighbors(i):
                    if state.board[neighbor] == Player.NONE:
                        mobility += 1
        return mobility

    def _count_blocked_positions(self, state: BoardState, player: Player) -> int:
        """Count the number of player's pieces that cannot move."""
        blocked = 0
        for i in range(state.board_size):
            if state.board[i] == player:
                is_blocked = True
                for neighbor in self.board.graph.iterNeighbors(i):
                    if state.board[neighbor] == Player.NONE:
                        is_blocked = False
                        break
                if is_blocked:
                    blocked += 1
        return blocked

    def _count_double_mill_configurations(self, state: BoardState, player: Player) -> int:
        """Count configurations where moving a piece can form multiple mills."""
        double_mills = 0

        # Check each player's piece
        for pos in range(state.board_size):
            if state.board[pos] == player:
                # For each empty neighbor
                for neighbor in self.board.graph.iterNeighbors(pos):
                    if state.board[neighbor] == Player.NONE:
                        # Simulate moving this piece
                        potential_mills = 0
                        temp_state = state.copy()
                        temp_state.board[pos] = Player.NONE
                        temp_state.board[neighbor] = player

                        # Count mills that would be formed
                        for mill in self.board.mills:
                            if neighbor in mill:
                                if (temp_state.board[mill[0]] == player and
                                        temp_state.board[mill[1]] == player and
                                        temp_state.board[mill[2]] == player):
                                    potential_mills += 1

                        # If moving this piece forms more than one mill
                        if potential_mills > 1:
                            double_mills += 1
                            break  # Count only once per piece

        return double_mills

    def _evaluate_winning_position(self, state: BoardState, player: Player) -> float:
        """
        Evaluate whether the current position is winning in a 3 vs 3 endgame.
        Analyzes positions and mobility patterns that lead to wins.
        """
        opponent = player.opponent()

        # Check if player has significantly better mobility
        player_mobility = self._count_mobility(state, player)
        opponent_mobility = self._count_mobility(state, opponent)
        mobility_advantage = player_mobility - opponent_mobility

        # Check for trapped opponent pieces
        opponent_blocked = self._count_blocked_positions(state, opponent)

        # Check for alignment of player pieces that controls important intersections
        control_score = 0
        for i in range(state.board_size):
            if state.board[i] == player:
                control_score += self.position_importance[i]
            elif state.board[i] == opponent:
                control_score -= self.position_importance[i]

        # Calculate a winning score based on the above factors
        winning_score = mobility_advantage * 2 + opponent_blocked * 3 + control_score * 0.5

        return winning_score