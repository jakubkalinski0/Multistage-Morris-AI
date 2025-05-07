# --- START OF FILE engines/rl_dqn_agent.py ---

import random
import os
from typing import Optional, List, Tuple, Deque, Dict
from collections import deque
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game.board.Board import Board
from game.board.BoardState import BoardState
from game.util.Move import Move, MoveType
from game.util.Player import Player
from game.util.Position import Position

class ActionValueNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super(ActionValueNetwork, self).__init__()
        # Consider making these configurable or larger for bigger boards
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, num_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x)); x = F.relu(self.layer2(x)); return self.layer3(x)

class ReplayMemory:
    def __init__(self, capacity): self.memory = deque([], maxlen=capacity)
    def push(self, experience: Tuple): self.memory.append(experience)
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)



import random
import os
from typing import Optional, List, Tuple
import torch

from game.board.Board import Board
from game.board.BoardState import BoardState
from game.util.Move import Move, MoveType
from game.util.Player import Player
from game.util.Position import Position
from game.GamePhase import GamePhase

# Import the 6MM agent implementation
from engines.RL_DQN_Agent import RLDQNAgent as RLDQNAgent6mm
from game.board.SixMensMorrisBoard import SixMensMorrisBoard

class RLDQNAgent:
    """
    A composite agent for 9MM that uses two 6MM agents for decision making.
    The 9MM board is conceptually divided into two 6MM sub-boards with overlapping middle positions.
    """
    
    # First 6MM board (outer square + middle positions)
    FIRST_6MM_MAP = {
        # Outer positions
        0: 0, 1: 1, 2: 2,       # Top outer row
        9: 6, 14: 9,            # Side outer positions
        21: 13, 22: 14, 23: 15, # Bottom outer row
        
        # Middle positions - repeated in both mappings
        3: 3, 4: 4, 5: 5,       # Middle top row
        10: 7, 13: 8,           # Middle side positions
        18: 10, 19: 11, 20: 12  # Middle bottom row
    }
    
    # Second 6MM board (inner square + middle positions)
    SECOND_6MM_MAP = {
        # Inner positions
        6: 0, 7: 1, 8: 2,       # Top inner row
        11: 6, 12: 9,           # Side inner positions
        15: 13, 16: 14, 17: 15, # Bottom inner row
        
        # Middle positions - repeated in both mappings
        3: 3, 4: 4, 5: 5,       # Middle top row
        10: 7, 13: 8,           # Middle side positions
        18: 10, 19: 11, 20: 12  # Middle bottom row
    }
    
    # Reverse mappings
    FIRST_6MM_TO_9MM = {v: k for k, v in FIRST_6MM_MAP.items()}
    SECOND_6MM_TO_9MM = {v: k for k, v in SECOND_6MM_MAP.items()}
    
    def __init__(self, board: Board, model_path: Optional[str] = None, device: Optional[str] = None, **kwargs):
        self.board = board
        self.board_size = board.board_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"RLDQNAgent9mm using two 6MM agents on device: {self.device}")
        
        # Create a dummy 6MM board for the agents
        self.dummy_6mm_board = SixMensMorrisBoard()
        
        # Path to 6mm model
        model_path_6mm = os.path.join("models", "6mm_rl_dqn_model.pth") if model_path is None else model_path.replace("9mm", "6mm")
        
        # Initialize two 6MM agents
        self.first_agent = RLDQNAgent6mm(self.dummy_6mm_board, model_path=model_path_6mm, device=device, **kwargs)
        self.second_agent = RLDQNAgent6mm(self.dummy_6mm_board, model_path=model_path_6mm, device=device, **kwargs)
        
        print(f"Initialized composite RLDQNAgent9mm with two 6MM agents using model: {model_path_6mm}")
        
        # For training purposes
        self.steps_done = 0
        self.training_episodes_count = 0

    def _map_9mm_to_6mm_board(self, state: BoardState, agent_number: int) -> BoardState:
        """Convert a 9MM board state to a 6MM board state for the specified agent."""
        # Create a new 6MM board state
        mapped_state = BoardState(self.dummy_6mm_board.board_size, self.dummy_6mm_board.pieces_per_player)
        
        # Copy global state properties
        mapped_state.current_player = state.current_player
        mapped_state.need_to_remove_piece = state.need_to_remove_piece
        
        # Set pieces count properties - divide by 2 due to 9/6 ratio, with overlapping positions
        mapped_state.pieces_left_to_place_by_player = {
            Player.WHITE: max(0, state.pieces_left_to_place_by_player[Player.WHITE] // 2),
            Player.BLACK: max(0, state.pieces_left_to_place_by_player[Player.BLACK] // 2)
        }
        
        # Map position mapping based on which agent
        pos_map = self.FIRST_6MM_MAP if agent_number == 1 else self.SECOND_6MM_MAP
        
        # Copy piece positions
        for pos_9mm, pos_6mm in pos_map.items():
            if pos_9mm < len(state.board):
                player = state.get_player_at_position(pos_9mm)
                if player != Player.NONE:
                    mapped_state.board[pos_6mm] = player
                    # Update piece count
                    mapped_state.pieces_from_player_currently_on_board[player] += 1
        
        # Determine game phase based on pieces on board and to place
        for player in [Player.WHITE, Player.BLACK]:
            if mapped_state.pieces_left_to_place_by_player[player] == 0:
                if mapped_state.pieces_from_player_currently_on_board[player] <= 3:
                    mapped_state.game_phase_for_player[player] = GamePhase.FLYING
                else:
                    mapped_state.game_phase_for_player[player] = GamePhase.MOVEMENT
            else:
                mapped_state.game_phase_for_player[player] = GamePhase.PLACEMENT
        
        return mapped_state

    def _map_move_between_boards(self, move: Move, from_9mm_to_6mm: bool, agent_number: int) -> Optional[Move]:
        """Map a move between 9MM and 6MM coordinates."""
        if move is None:
            return None
            
        pos_map = self.FIRST_6MM_MAP if agent_number == 1 else self.SECOND_6MM_MAP
        reverse_map = self.FIRST_6MM_TO_9MM if agent_number == 1 else self.SECOND_6MM_TO_9MM
        
        # Select the appropriate mapping direction
        mapping = pos_map if from_9mm_to_6mm else reverse_map
        
        if move.move_type == MoveType.PLACE:
            pos_id = move.to_position.id
            if pos_id in mapping:
                return Move(
                    move_type=MoveType.PLACE,
                    to_position=Position(mapping[pos_id])
                )
        elif move.move_type == MoveType.MOVE:
            from_id = move.from_position.id
            to_id = move.to_position.id
            if from_id in mapping and to_id in mapping:
                return Move(
                    move_type=MoveType.MOVE,
                    from_position=Position(mapping[from_id]),
                    to_position=Position(mapping[to_id])
                )
        elif move.move_type == MoveType.REMOVE:
            pos_id = move.remove_checker_from_position.id
            if pos_id in mapping:
                return Move(
                    move_type=MoveType.REMOVE,
                    remove_checker_from_position=Position(mapping[pos_id])
                )
        return None

    def _filter_legal_moves(self, moves: List[Move], agent_number: int) -> List[Move]:
        """Filter legal moves to only include those in the agent's region."""
        pos_map = self.FIRST_6MM_MAP if agent_number == 1 else self.SECOND_6MM_MAP
        valid_positions = set(pos_map.keys())
        
        filtered_moves = []
        for move in moves:
            if move.move_type == MoveType.PLACE:
                if move.to_position.id in valid_positions:
                    filtered_moves.append(move)
            elif move.move_type == MoveType.MOVE:
                if move.from_position.id in valid_positions and move.to_position.id in valid_positions:
                    filtered_moves.append(move)
            elif move.move_type == MoveType.REMOVE:
                if move.remove_checker_from_position.id in valid_positions:
                    filtered_moves.append(move)
        
        return filtered_moves

    def get_best_move(self, state: BoardState) -> Optional[Move]:
        """Get the best move using both 6MM agents."""
        legal_moves = self.board.get_legal_moves(state)
        if not legal_moves:
            return None
            
        # Get legal moves for each region
        region1_moves = self._filter_legal_moves(legal_moves, 1)
        region2_moves = self._filter_legal_moves(legal_moves, 2)
        
        # No moves in a region? Use the other one exclusively
        if not region1_moves:
            return self._get_best_move_from_region(state, 2, region2_moves)
        if not region2_moves:
            return self._get_best_move_from_region(state, 1, region1_moves)
        
        # Get the best move from each agent with Q-value
        move1, q1 = self._get_move_with_q_value(state, 1, region1_moves)
        move2, q2 = self._get_move_with_q_value(state, 2, region2_moves)
        
        # Compare Q-values and choose the better move
        if q1 >= q2:
            return move1
        else:
            return move2

    def _get_best_move_from_region(self, state: BoardState, agent_number: int, filtered_moves: List[Move]) -> Optional[Move]:
        """Get best move from a specific region."""
        if not filtered_moves:
            return None
            
        # Map the 9MM state to a 6MM state
        mapped_state = self._map_9mm_to_6mm_board(state, agent_number)
        agent = self.first_agent if agent_number == 1 else self.second_agent
        
        # Map each legal move to 6MM coordinates
        mapped_moves_dict = {}  # 6MM move string -> 9MM move
        for move in filtered_moves:
            mapped_move = self._map_move_between_boards(move, True, agent_number)
            if mapped_move:
                mapped_moves_dict[str(mapped_move)] = move
        
        # If no valid mapped moves, fall back to random
        if not mapped_moves_dict:
            return random.choice(filtered_moves)
        
        # Get best move from agent in 6MM coordinates 
        best_6mm_move = agent.get_best_move(mapped_state)
        
        # Map back to 9MM move
        if best_6mm_move and str(best_6mm_move) in mapped_moves_dict:
            return mapped_moves_dict[str(best_6mm_move)]
        else:
            # Fallback to random if mapping fails
            return random.choice(filtered_moves)

    def _get_move_with_q_value(self, state: BoardState, agent_number: int, filtered_moves: List[Move]) -> Tuple[Optional[Move], float]:
        """Get the best move from an agent along with its Q-value."""
        if not filtered_moves:
            return None, -float('inf')
            
        # Map the 9MM state to a 6MM state
        mapped_state = self._map_9mm_to_6mm_board(state, agent_number)
        agent = self.first_agent if agent_number == 1 else self.second_agent
        
        try:
            # Map each legal move to 6MM coordinates
            mapped_moves = []
            original_moves = []
            for move in filtered_moves:
                mapped_move = self._map_move_between_boards(move, True, agent_number)
                if mapped_move:
                    mapped_moves.append(mapped_move)
                    original_moves.append(move)
            
            # If no valid mapped moves, fall back to random with low Q-value
            if not mapped_moves:
                return random.choice(filtered_moves), -100.0
            
            # Get state tensor
            state_tensor = agent._state_to_tensor(mapped_state)
            
            # Get Q-values for all actions
            with torch.no_grad():
                q_values = agent.policy_net(state_tensor)[0]
                
            # Find move with highest Q-value
            best_q_val = -float('inf')
            best_move = None
            
            for i, mapped_move in enumerate(mapped_moves):
                idx = agent._get_action_index(mapped_move)
                if idx is not None and 0 <= idx < agent.num_actions:
                    q_val = q_values[idx].item()
                    if q_val > best_q_val:
                        best_q_val = q_val
                        best_move = original_moves[i]
            
            if best_move:
                return best_move, best_q_val
            else:
                # Fall back to random if no valid moves found
                return random.choice(filtered_moves), -10.0
                
        except Exception as e:
            print(f"Error getting Q-value for agent {agent_number}: {e}")
            return random.choice(filtered_moves), -50.0

    # Additional methods for compatibility with tournament system
    def select_action_epsilon_greedy(self, state: BoardState):
        return self.get_best_move(state), None
        
    def save_model(self, path: Optional[str] = None):
        # Just for compatibility - doesn't save anything new
        print("Note: RLDQNAgent9mm doesn't save - it uses the 6mm model")
        
    def optimize_model(self):
        # Dummy method for compatibility
        pass

# --- END OF FILE engines/rl_dqn_agent.py ---