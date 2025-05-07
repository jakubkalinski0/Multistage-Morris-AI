# --- START OF FILE engines/mdqn_agent.py ---
# (Importy, MDQN_SubNetwork, ReplayMemory bez zmian z ostatniej pełnej wersji)
import random
import os
from typing import Optional, List, Tuple, Deque, Dict
from collections import deque
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from game.board.Board import Board
from game.board.BoardState import BoardState
from game.util.Move import Move, MoveType
from game.util.Player import Player
from game.util.Position import Position
from game.GamePhase import GamePhase

# Import the 6MM agent implementation
from engines.mDQN_Agent import MDQNAgent as MDQNAgent6mm
from game.board.SixMensMorrisBoard import SixMensMorrisBoard

class MDQN_SubNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super(MDQN_SubNetwork, self).__init__()
        # Consider making these configurable or larger for bigger boards
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, num_actions)
    def forward(self, x):
        x = F.relu(self.layer1(x)); x = F.relu(self.layer2(x)); return self.layer3(x)

class ReplayMemory:
    def __init__(self, capacity): self.memory = deque([], maxlen=capacity)
    def push(self, experience: Tuple): self.memory.append(experience)
    def sample(self, batch_size): return random.sample(self.memory, batch_size)
    def __len__(self): return len(self.memory)


class BaseSubAgent:
    DEFAULT_INPUT_SIZE_3MM = 9 * 3 + 1 + 2 + 2 + 1

    def __init__(self, board_size: int, num_specific_actions: int, model_filename_suffix: str,
                 device: torch.device, learning_rate, gamma, epsilon_start, epsilon_end,
                 epsilon_decay, target_update_freq, replay_memory_capacity, batch_size,
                 model_path_prefix: str = "models"):
        self.board_size = board_size
        self.input_size = self.board_size * 3 + 1 + 2 + 2 + 1
        self.num_actions = num_specific_actions
        self.device = device
        self.model_filename_suffix = model_filename_suffix
        # Construct model path using board_size and suffix, relative to prefix
        self.model_path_from_init = os.path.join(model_path_prefix, f"{self.board_size}mm_mdqn_{model_filename_suffix}.pth")

        self.policy_net = MDQN_SubNetwork(self.input_size, self.num_actions).to(device)
        self.target_net = MDQN_SubNetwork(self.input_size, self.num_actions).to(device)
        self.model_loaded = False

        model_dir = os.path.dirname(self.model_path_from_init)
        if model_dir and not os.path.exists(model_dir): # Ensure directory exists
            try: os.makedirs(model_dir, exist_ok=True)
            except OSError as e: print(f"Could not create model dir {model_dir}: {e}")

        if os.path.exists(self.model_path_from_init):
            try:
                self.policy_net.load_state_dict(torch.load(self.model_path_from_init, map_location=device))
                self.model_loaded = True
                print(f"Loaded model for {model_filename_suffix} ({self.board_size}MM) from {self.model_path_from_init}")
            except Exception as e: print(f"Error loading {model_filename_suffix} ({self.board_size}MM) model: {e}. Fresh weights.")
        else: print(f"{model_filename_suffix} ({self.board_size}MM) model not found at {self.model_path_from_init}. Fresh weights.")

        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.memory = ReplayMemory(replay_memory_capacity)
        self.gamma = gamma; self.epsilon_start = epsilon_start; self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay; self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0

    # ... (Reszta metod BaseSubAgent: _state_to_tensor, optimize_model_step, ...)
    # ... (update_target_network_if_needed, save_model_sub - takie same jak w ostatniej pełnej wersji) ...
    def _state_to_tensor(self, state: BoardState) -> torch.Tensor:
        board_repr = []
        for i in range(self.board_size):
            player = state.get_player_at_position(i)
            if player == Player.NONE: board_repr.extend([1.0, 0.0, 0.0])
            elif player == Player.WHITE: board_repr.extend([0.0, 1.0, 0.0])
            else: board_repr.extend([0.0, 0.0, 1.0])
        cp_repr = [1.0] if state.current_player == Player.WHITE else [-1.0]
        pl_w = float(state.pieces_left_to_place_by_player[Player.WHITE]); pl_b = float(state.pieces_left_to_place_by_player[Player.BLACK])
        po_w = float(state.pieces_from_player_currently_on_board[Player.WHITE]); po_b = float(state.pieces_from_player_currently_on_board[Player.BLACK])
        rm_r = [1.0] if state.need_to_remove_piece else [0.0]
        features = board_repr + cp_repr + [pl_w, pl_b] + [po_w, po_b] + rm_r
        if len(features) != self.input_size:
            raise ValueError(f"Feature length mismatch: expected {self.input_size}, got {len(features)}")
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

    def optimize_model_step(self):
        if len(self.memory) < self.batch_size: return
        experiences = self.memory.sample(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch_list, done_batch_list = zip(*experiences)
        state_batch_tensor = torch.cat(state_batch).to(self.device)
        action_batch_tensor = torch.tensor(action_batch, dtype=torch.long).unsqueeze(1).to(self.device)
        reward_batch_tensor = torch.tensor(reward_batch, dtype=torch.float32).unsqueeze(1).to(self.device)
        done_batch_tensor = torch.tensor(done_batch_list, dtype=torch.bool).unsqueeze(1).to(self.device)
        non_final_mask = torch.tensor(tuple(s is not None for s in next_state_batch_list), dtype=torch.bool).to(self.device)
        non_final_next_states_list = [s for s in next_state_batch_list if s is not None]
        current_q_values = self.policy_net(state_batch_tensor).gather(1, action_batch_tensor)
        next_state_q_values = torch.zeros(self.batch_size, 1, device=self.device)
        if len(non_final_next_states_list) > 0:
            non_final_next_states_tensor = torch.cat(non_final_next_states_list).to(self.device)
            with torch.no_grad():
                next_state_q_values[non_final_mask] = self.target_net(non_final_next_states_tensor).max(1, keepdim=True)[0]
        expected_q_values = reward_batch_tensor + (self.gamma * next_state_q_values * (~done_batch_tensor))
        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q_values, expected_q_values)
        self.optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_network_if_needed(self, global_episode_count): # Parameter passed from MDQNAgent
        if global_episode_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model_sub(self):
        os.makedirs(os.path.dirname(self.model_path_from_init), exist_ok=True)
        torch.save(self.policy_net.state_dict(), self.model_path_from_init)
        print(f"{self.model_filename_suffix} ({self.board_size}MM) model saved to {self.model_path_from_init}")


# --- Specific Sub-Agents (PlacementAgentDQN, MovementAgentDQN, RemovalAgentDQN - same as before) ---
class PlacementAgentDQN(BaseSubAgent):
    def __init__(self, board_size, device, **kwargs): super().__init__(board_size, board_size, "placement", device, **kwargs)
    def get_action_index(self, move: Move) -> Optional[int]:
        if move.move_type == MoveType.PLACE: return move.to_position.id
        return None
    def get_move_from_index(self, action_index: int) -> Optional[Move]:
        if 0 <= action_index < self.board_size: return Move(MoveType.PLACE, to_position=Position(action_index))
        return None

class MovementAgentDQN(BaseSubAgent):
    def __init__(self, board_size, device, **kwargs):
        super().__init__(board_size, board_size * (board_size - 1), "movement", device, **kwargs)
    def get_action_index(self, move: Move) -> Optional[int]:
        if move.move_type == MoveType.MOVE:
            from_id=move.from_position.id; to_id=move.to_position.id
            to_idx_adj = to_id if to_id < from_id else to_id - 1
            return from_id * (self.board_size - 1) + to_idx_adj
        return None
    def get_move_from_index(self, action_index: int) -> Optional[Move]:
        if 0 <= action_index < self.num_actions:
            from_id = action_index // (self.board_size - 1)
            to_idx_adj = action_index % (self.board_size - 1)
            to_id = to_idx_adj if to_idx_adj < from_id else to_idx_adj + 1
            return Move(MoveType.MOVE, from_position=Position(from_id), to_position=Position(to_id))
        return None

class RemovalAgentDQN(BaseSubAgent):
    def __init__(self, board_size, device, **kwargs):
        super().__init__(board_size, board_size, "removal", device, **kwargs)
    def get_action_index(self, move: Move) -> Optional[int]:
        if move.move_type == MoveType.REMOVE: return move.remove_checker_from_position.id
        return None
    def get_move_from_index(self, action_index: int) -> Optional[Move]:
        if 0 <= action_index < self.board_size: return Move(MoveType.REMOVE, remove_checker_from_position=Position(action_index))
        return None

# --- Main MDQN Agent (Dispatcher - same as before) ---
class MDQNAgent:
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
    
    def __init__(self, board: Board, device: Optional[str] = None, **hyperparams):
        self.board = board
        self.board_size = board.board_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"MDQNAgent9mm using two 6MM agents on device: {self.device}")
        
        # Create a dummy 6MM board for the agents
        self.dummy_6mm_board = SixMensMorrisBoard()
        
        # Initialize two 6MM agents
        self.first_agent = MDQNAgent6mm(self.dummy_6mm_board, device=device, **hyperparams)
        self.second_agent = MDQNAgent6mm(self.dummy_6mm_board, device=device, **hyperparams)
        
        print("Initialized composite MDQNAgent9mm with two 6MM agents")
        
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
            # Get the sub-agent for current game phase
            agent_key, sub_agent = agent._get_current_sub_agent_key_and_instance(mapped_state)
            
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
            state_tensor = sub_agent._state_to_tensor(mapped_state)
            
            # Get Q-values for valid moves
            with torch.no_grad():
                q_values = sub_agent.policy_net(state_tensor)[0]
                
            # Find move with highest Q-value
            best_q_val = -float('inf')
            best_move = None
            
            for i, mapped_move in enumerate(mapped_moves):
                idx = sub_agent.get_action_index(mapped_move)
                if idx is not None and 0 <= idx < sub_agent.num_actions:
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

    # Tournament compatibility methods
    def select_action_epsilon_greedy(self, state: BoardState):
        """For tournament compatibility."""
        return self.get_best_move(state), None
        
    def update_target_networks_globally(self):
        """Update target networks for both agents."""
        self.first_agent.update_target_networks_globally()
        self.second_agent.update_target_networks_globally()
        self.training_episodes_count += 1
        
    def save_models(self):
        """Save both 6MM agent models."""
        self.first_agent.save_models()
        self.second_agent.save_models()
        print("Both 6MM agents saved their models.")
        
    def push_experience(self, *args):
        """For tournament compatibility."""
        pass
        
    def optimize_sub_agent(self, *args):
        """For tournament compatibility."""
        pass

# --- END OF FILE engines/mdqn_agent.py ---