# --- START OF FILE engines/mdqn_agent.py ---
# (Importy, MDQN_SubNetwork, ReplayMemory bez zmian z ostatniej pełnej wersji)
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
from game.GamePhase import GamePhase

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
    def __init__(self, board: Board, device: Optional[str] = None, **hyperparams):
        self.board = board; self.board_size = board.board_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"MDQNAgent for {self.board_size}MM using device: {self.device}")
        default_hyperparams = {
            'learning_rate':0.00025, 'gamma':0.99, 'epsilon_start':1.0, 'epsilon_end':0.05,
            'epsilon_decay':30000, 'target_update_freq':20, 'replay_memory_capacity':30000, 'batch_size':128
        }
        current_hyperparams = {**default_hyperparams, **hyperparams}
        # Pass the root model directory prefix
        current_hyperparams['model_path_prefix'] = "models" # Sub-agents will use this

        self.placement_agent = PlacementAgentDQN(self.board_size, self.device, **current_hyperparams)
        self.movement_agent = MovementAgentDQN(self.board_size, self.device, **current_hyperparams)
        self.removal_agent = RemovalAgentDQN(self.board_size, self.device, **current_hyperparams)
        self.sub_agents: Dict[str, BaseSubAgent] = {
            "placement": self.placement_agent, "movement": self.movement_agent, "removal": self.removal_agent
        }
        self.steps_done = 0; self.training_episodes_count = 0

    # ... (Reszta metod MDQNAgent: _get_current_sub_agent_key_and_instance, select_action_epsilon_greedy, ...)
    # ... (get_best_move, optimize_sub_agent, update_target_networks_globally, save_models, push_experience) ...
    # --- Te metody pozostają takie same jak w OSTATNIEJ PEŁNEJ WERSJI MDQNAgent ---
    def _get_current_sub_agent_key_and_instance(self, state: BoardState) -> Tuple[str, BaseSubAgent]:
        if state.need_to_remove_piece: return "removal", self.removal_agent
        current_phase = state.get_current_phase_for_player(state.current_player)
        if current_phase == GamePhase.PLACEMENT: return "placement", self.placement_agent
        elif current_phase in [GamePhase.MOVEMENT, GamePhase.FLYING]: return "movement", self.movement_agent
        raise ValueError("Unknown game state for selecting sub-agent")

    def select_action_epsilon_greedy(self, state: BoardState) -> Tuple[Optional[Move], Optional[int]]:
        _, sub_agent = self._get_current_sub_agent_key_and_instance(state)
        sub_agent.steps_done = self.steps_done
        legal_moves = self.board.get_legal_moves(state)
        if not legal_moves: return None, None
        eps_threshold = sub_agent.epsilon_end + (sub_agent.epsilon_start - sub_agent.epsilon_end) * \
                        np.exp(-1. * self.steps_done / sub_agent.epsilon_decay)
        if random.random() < eps_threshold or not sub_agent.model_loaded:
            chosen_move = random.choice(legal_moves)
            action_idx = sub_agent.get_action_index(chosen_move)
            if action_idx is None:
                for m in legal_moves:
                    idx_try = sub_agent.get_action_index(m)
                    if idx_try is not None: chosen_move = m; action_idx = idx_try; break
                if action_idx is None: return None, None
            return chosen_move, action_idx
        else: # Exploit
            with torch.no_grad():
                state_tensor = sub_agent._state_to_tensor(state)
                q_values = sub_agent.policy_net(state_tensor)[0]
                best_q_val = -float('inf'); best_move = None; best_idx = None
                for move in legal_moves:
                    idx = sub_agent.get_action_index(move)
                    if idx is not None and 0 <= idx < sub_agent.num_actions:
                        if q_values[idx].item() > best_q_val:
                            best_q_val = q_values[idx].item(); best_move = move; best_idx = idx
                if best_move: return best_move, best_idx
                else: chosen_move = random.choice(legal_moves); return chosen_move, sub_agent.get_action_index(chosen_move)

    def get_best_move(self, state: BoardState) -> Optional[Move]: # For gameplay
        _, sub_agent = self._get_current_sub_agent_key_and_instance(state)
        legal_moves = self.board.get_legal_moves(state)
        if not legal_moves: return None
        if not sub_agent.model_loaded: return random.choice(legal_moves)
        with torch.no_grad():
            state_tensor = sub_agent._state_to_tensor(state)
            q_values = sub_agent.policy_net(state_tensor)[0]
            best_q_val = -float('inf'); chosen_move = None
            for move in legal_moves:
                idx = sub_agent.get_action_index(move)
                if idx is not None and 0 <= idx < sub_agent.num_actions:
                    if q_values[idx].item() > best_q_val:
                        best_q_val = q_values[idx].item(); chosen_move = move
            if chosen_move: return chosen_move
            else: return random.choice(legal_moves)

    def optimize_sub_agent(self, sub_agent_key: str): self.sub_agents[sub_agent_key].optimize_model_step()
    def update_target_networks_globally(self):
        self.training_episodes_count += 1
        for _, agent_instance in self.sub_agents.items():
            agent_instance.update_target_network_if_needed(self.training_episodes_count)
    def save_models(self):
        for _, agent_instance in self.sub_agents.items(): agent_instance.save_model_sub()
        print("All mDQN sub-agent models saved.")
    def push_experience(self, state_tensor, action_idx, reward, next_state_tensor, done, state_for_sub_agent_check: BoardState):
        _, sub_agent = self._get_current_sub_agent_key_and_instance(state_for_sub_agent_check)
        sub_agent.memory.push((state_tensor, action_idx, reward, next_state_tensor, done))


# --- END OF FILE engines/mdqn_agent.py ---