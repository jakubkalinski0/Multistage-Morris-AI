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
# from game.GamePhase import GamePhase # Not directly used

class ActionValueNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super(ActionValueNetwork, self).__init__()
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

class RLDQNAgent:
    DEFAULT_INPUT_SIZE_3MM = 9 * 3 + 1 + 2 + 2 + 1

    def __init__(self, board: Board, model_path: Optional[str] = "models/3mm_rl_dqn_model.pth",
                 device: Optional[str] = None, learning_rate=0.00025, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=30000,
                 target_update_freq=20, replay_memory_capacity=50000, batch_size=128):
        self.board = board; self.board_size = board.board_size
        self.model_path_from_init = model_path
        self.input_size = self.board_size * 3 + 1 + 2 + 2 + 1
        self.num_actions = self._get_max_possible_actions_for_board()
        if device: self.device = torch.device(device)
        else: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"RLDQNAgent using device: {self.device}")
        self.policy_net = ActionValueNetwork(self.input_size, self.num_actions).to(self.device)
        self.target_net = ActionValueNetwork(self.input_size, self.num_actions).to(self.device)
        self.model_loaded = False
        if model_path:
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                try: os.makedirs(model_dir, exist_ok=True)
                except OSError as e: print(f"Could not create model dir: {e}")
            if os.path.exists(model_path):
                try:
                    self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.model_loaded = True; print(f"Loaded RL DQN model: {model_path}")
                except Exception as e: print(f"Error loading RL DQN model: {e}. Fresh weights.")
            else: print(f"RL DQN model not found: {model_path}. Fresh weights.")
        else: print("No model path for RL DQN. Fresh weights.")
        self.target_net.load_state_dict(self.policy_net.state_dict()); self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        self.memory = ReplayMemory(replay_memory_capacity)
        self.gamma = gamma; self.epsilon_start = epsilon_start; self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay; self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0; self.training_episodes_count = 0
        self.action_to_move_map: Dict[int, Move] = {}; self.move_to_action_map: Dict[Move, int] = {}
        self._initialize_action_space_map()

    def _get_max_possible_actions_for_board(self):
        return self.board_size + self.board_size + self.board_size * (self.board_size - 1)

    def _initialize_action_space_map(self):
        idx = 0
        for i in range(self.board_size): # Place
            move = Move(MoveType.PLACE, to_position=Position(i))
            self.action_to_move_map[idx] = move; self.move_to_action_map[move] = idx; idx += 1
        for i in range(self.board_size): # Remove
            move = Move(MoveType.REMOVE, remove_checker_from_position=Position(i))
            self.action_to_move_map[idx] = move; self.move_to_action_map[move] = idx; idx += 1
        for r in range(self.board_size): # Move/Fly
            for c in range(self.board_size):
                if r == c: continue
                move = Move(MoveType.MOVE, from_position=Position(r), to_position=Position(c))
                self.action_to_move_map[idx] = move; self.move_to_action_map[move] = idx; idx += 1
        if idx != self.num_actions: print(f"CRITICAL: Action space init mismatch. Expected {self.num_actions}, got {idx}.")

    def _get_action_index(self, move: Move) -> Optional[int]: return self.move_to_action_map.get(move)
    def _get_move_from_index(self, action_index: int) -> Optional[Move]: return self.action_to_move_map.get(action_index)

    def _state_to_tensor(self, state: BoardState) -> torch.Tensor: # (Same as StateValueAgent)
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

    def select_action_epsilon_greedy(self, state: BoardState) -> Tuple[Optional[Move], Optional[int]]:
        legal_moves = self.board.get_legal_moves(state)
        if not legal_moves: return None, None
        eps_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                        np.exp(-1. * self.steps_done / self.epsilon_decay)
        # self.steps_done is incremented by the main training loop

        if random.random() < eps_threshold or not self.model_loaded:
            chosen_move = random.choice(legal_moves)
            action_idx = self._get_action_index(chosen_move)
            if action_idx is None:
                # Fallback if a legal move from board logic isn't in our predefined map
                # This might happen if the action space is too restrictive or mapping has holes.
                # print(f"Warning: Random legal move {chosen_move} not in action map during exploration. Trying first legal.")
                for m_idx, m_obj in self.action_to_move_map.items(): # Try to find any valid mapped legal move
                    if m_obj in legal_moves: chosen_move = m_obj; action_idx = m_idx; break
                if action_idx is None: return None, None # Serious issue
            return chosen_move, action_idx
        else: # Exploit
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state)
                q_values_all = self.policy_net(state_tensor)[0]
                best_q_val = -float('inf'); chosen_move = None; action_idx = None
                for move in legal_moves:
                    idx = self._get_action_index(move)
                    if idx is not None and 0 <= idx < self.num_actions:
                        if q_values_all[idx].item() > best_q_val:
                            best_q_val = q_values_all[idx].item(); chosen_move = move; action_idx = idx
                if chosen_move: return chosen_move, action_idx
                else: fallback_move = random.choice(legal_moves); return fallback_move, self._get_action_index(fallback_move)

    def get_best_move(self, state: BoardState) -> Optional[Move]: # For gameplay
        # Essentially the same as exploitation part of select_action_epsilon_greedy
        legal_moves = self.board.get_legal_moves(state)
        if not legal_moves: return None
        if not self.model_loaded: return random.choice(legal_moves)
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values_all = self.policy_net(state_tensor)[0]
            best_q_val = -float('inf'); chosen_move = None
            for move in legal_moves:
                action_idx = self._get_action_index(move)
                if action_idx is not None and 0 <= action_idx < self.num_actions:
                    q_val = q_values_all[action_idx].item()
                    if q_val > best_q_val: best_q_val = q_val; chosen_move = move
            if chosen_move: return chosen_move
            else: return random.choice(legal_moves)

    def optimize_model(self): # (Remains the same as previous correct version)
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

    def update_target_network_if_needed(self):
        self.training_episodes_count +=1 # Increment here, called once per episode from training script
        if self.training_episodes_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            # print(f"RLDQN Target network updated at episode {self.training_episodes_count}")

    def save_model(self, path: Optional[str] = None):
        save_path = path if path else self.model_path_from_init
        if not save_path:
            save_path = os.path.join("models", f"rl_dqn_{self.board_size}mm_ep{self.training_episodes_count}.pth")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"RL DQN model saved to {save_path}")

# --- END OF FILE engines/rl_dqn_agent.py ---