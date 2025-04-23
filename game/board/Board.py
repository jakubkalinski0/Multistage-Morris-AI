from abc import ABC, abstractmethod
from typing import List, Tuple

import networkit as nk

from game.board.BoardState import BoardState
from game.util.Move import Move
from game.util.MoveType import MoveType
from game.util.Player import Player
from game.util.Position import Position
from game.GamePhase import GamePhase


class Board(ABC):
    def __init__(self, board_graph: Tuple[Tuple[int]], pieces_per_player: int, board_mills):
        self.board_size = len(board_graph)
        self.graph = nk.graph.Graph(self.board_size, directed=False)
        self.pieces_per_player = pieces_per_player
        self.mills = board_mills
        self.initialize_connections(board_graph)
        self.initialize_board()

    def initialize_connections(self, board_graph):
        for v, k in enumerate(board_graph):
            for i in k:
                self.graph.addEdge(v, i)

    def initialize_board(self):
        self.board_state = self.get_initial_board_state()

    def get_initial_board_state(self) -> BoardState:
        return BoardState(self.board_size, self.pieces_per_player)

    def mill_check(self, state: BoardState, position: int, mill: Tuple[int], player: Player) -> bool:
        """Check if a mill is formed with the given position and player."""
        if position == mill[0] and player == state.board[mill[1]] and player == state.board[mill[2]]:
            return True
        if position == mill[1] and player == state.board[mill[0]] and player == state.board[mill[2]]:
            return True
        if position == mill[2] and player == state.board[mill[0]] and player == state.board[mill[1]]:
            return True
        return False

    def check_if_move_creates_mill(self, state: BoardState, position: int, player: Player) -> int:
        """Return the number of mills created by placing a piece at the given position."""
        mills_counter = 0
        for mill in self.mills:
            if position in mill:
                if self.mill_check(state, position, mill, player):
                    mills_counter += 1

        return mills_counter

    def get_legal_moves(self, state: BoardState) -> List[Move]:
        """Return all legal moves for the current player in the given state."""
        legal_moves = []
        current_player = state.current_player
        phase = state.get_current_phase_for_player(current_player)

        if state.need_to_remove_piece:
            # Return moves to remove opponent's pieces
            opponent = current_player.opponent()
            for i in range(self.board_size):
                if state.get_player_at_position(i) == opponent:
                    # Can't remove pieces that form a mill unless all opponent pieces form mills
                    if not self.check_if_move_creates_mill(state, i, opponent) or self.all_pieces_form_mills(state, opponent):
                        legal_moves.append(Move(MoveType.REMOVE, None, None, Position(i)))
            return legal_moves

        if phase == GamePhase.PLACEMENT:
            # Return moves to place a piece
            for i in range(self.board_size):
                if state.is_position_empty(i):
                    legal_moves.append(Move(MoveType.PLACE, None, Position(i)))

        elif phase == GamePhase.MOVEMENT:
            # Return moves to move a piece
            for i in range(self.board_size):
                if state.get_player_at_position(i) == current_player:
                    for j in self.graph.iterNeighbors(i):
                        if state.is_position_empty(j):
                            legal_moves.append(Move(MoveType.MOVE, Position(i), Position(j)))

        elif phase == GamePhase.FLYING:
            # Return moves to fly a piece (move to any empty position)
            for i in range(self.board_size):
                if state.get_player_at_position(i) == current_player:
                    for j in range(self.board_size):
                        if state.is_position_empty(j):
                            legal_moves.append(Move(MoveType.MOVE, Position(i), Position(j)))

        return legal_moves

    def all_pieces_form_mills(self, state: BoardState, player: Player) -> bool:
        """Check if all pieces of the given player form mills."""
        for i in range(self.board_size):
            if state.get_player_at_position(i) == player:
                # If this piece doesn't form a mill, return False
                if not self.check_if_move_creates_mill(state, i, player):
                    return False
        return True

    def make_move(self, state: BoardState, move: Move) -> BoardState:
        """Update the board state after making a move."""
        new_state = state.copy() # Robię tu kopie bo zakładam że będziemy chcieli robić graf stanów i lepiej operować na kopiach dla ewaluacji imo
        player = new_state.current_player

        if move.move_type == MoveType.PLACE:
            # Place a piece on the board
            position = move.to_position.id
            new_state.board[position] = player
            new_state.pieces_left_to_place_by_player[player] -= 1
            new_state.pieces_from_player_currently_on_board[player] += 1

            # Update game phase if all pieces are placed
            if new_state.pieces_left_to_place_by_player[player] == 0:
                new_state.game_phase_for_player[player] = GamePhase.MOVEMENT

            # Check if the move creates a mill
            mill_created = self.check_if_move_creates_mill(new_state, position, player) > 0
            new_state.need_to_remove_piece = mill_created

        elif move.move_type == MoveType.MOVE:
            # Move a piece on the board
            new_state.board[move.from_position.id] = Player.NONE
            new_state.board[move.to_position.id] = player

            # Check if the move creates a mill
            mill_created = self.check_if_move_creates_mill(new_state, move.to_position.id, player) > 0
            new_state.need_to_remove_piece = mill_created

        elif move.move_type == MoveType.REMOVE:
            # Remove an opponent's piece
            position = move.remove_checker_from_position.id
            opponent = player.opponent()
            new_state.board[position] = Player.NONE
            new_state.pieces_from_player_currently_on_board[opponent] -= 1

            # Check if opponent should enter flying phase
            if (new_state.pieces_from_player_currently_on_board[opponent] == 3 and
                    new_state.pieces_left_to_place_by_player[opponent] == 0 and
                    new_state.game_phase_for_player[opponent] == GamePhase.MOVEMENT):
                new_state.game_phase_for_player[opponent] = GamePhase.FLYING

            new_state.need_to_remove_piece = False

        # Update current player only if no piece needs to be removed
        if not new_state.need_to_remove_piece:
            new_state.current_player = player.opponent()

        # Add move to history
        new_state.move_history.append(move)

        return new_state

    def check_if_game_is_over(self, state: BoardState) -> bool:
        # Game is over if a player has fewer than 3 pieces after placement phase
        white_done_placing = state.pieces_left_to_place_by_player[Player.WHITE] == 0
        black_done_placing = state.pieces_left_to_place_by_player[Player.BLACK] == 0

        if white_done_placing and black_done_placing:
            if (state.pieces_from_player_currently_on_board[Player.WHITE] < 3 or
                    state.pieces_from_player_currently_on_board[Player.BLACK] < 3):
                return True

            # Also check if current player has no legal moves
            legal_moves = self.get_legal_moves(state)
            return len(legal_moves) == 0

        return False

    def get_winner(self, state: BoardState) -> Player:
        if state.pieces_from_player_currently_on_board[Player.WHITE]+state.pieces_left_to_place_by_player[Player.WHITE] < 3:
            return Player.BLACK
        elif state.pieces_from_player_currently_on_board[Player.BLACK]+state.pieces_left_to_place_by_player[Player.BLACK] < 3:
            return Player.WHITE

        # If current player has no legal moves, opponent wins
        if len(self.get_legal_moves(state)) == 0:
            return state.current_player.opponent()

        return Player.NONE  # Game not over or draw

    @abstractmethod
    def print_map_2d(self):
        """Print a 2D representation of the board."""
        pass