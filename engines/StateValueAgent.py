# --- START OF FILE engines/dqn_agent.py ---

import random
import os
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Use absolute imports from the project root perspective
from game.board.Board import Board
from game.board.BoardState import BoardState
from game.util.Move import Move
from game.util.Player import Player

# Define the Neural Network structure
class ValueNetwork(nn.Module): # Renamed to be more specific
    """Simple MLP to predict state value."""
    def __init__(self, input_size, output_size=1):
        super(ValueNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.tanh(self.layer3(x)) # Output between -1 and 1
        return x

class StateValueAgent:
    """
    Agent using a trained network to evaluate board states and choose moves.
    The network is trained via supervised learning on state values derived
    from an evaluated game graph.
    """
    DEFAULT_INPUT_SIZE_3MM = 9 * 3 + 1 + 2 + 2 + 1 # = 33

    def __init__(self, board: Board, model_path: Optional[str] = "models/3mm_state_value_model.pth", device: Optional[str] = None):
        self.board = board
        self.board_size = board.board_size
        self.input_size = self.board_size * 3 + 1 + 2 + 2 + 1

        if self.board_size != 9 and self.input_size != StateValueAgent.DEFAULT_INPUT_SIZE_3MM:
             print(f"Warning: StateValueAgent model trained for 3MM (input {StateValueAgent.DEFAULT_INPUT_SIZE_3MM}). "
                   f"Current board input {self.input_size}. Performance may vary.")

        if device: self.device = torch.device(device)
        else: self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"StateValueAgent using device: {self.device}")

        self.value_net = ValueNetwork(self.input_size).to(self.device)
        self.model_loaded = False

        if model_path:
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                try: os.makedirs(model_dir, exist_ok=True)
                except OSError as e: print(f"Could not create model directory {model_dir}: {e}")
            if os.path.exists(model_path):
                try:
                    self.value_net.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.value_net.eval()
                    self.model_loaded = True
                    print(f"Loaded StateValueAgent model from {model_path}")
                except Exception as e: print(f"Error loading StateValueAgent model: {e}. Random play.")
            else: print(f"StateValueAgent model not found: {model_path}. Random play.")
        else: print("No model path for StateValueAgent. Random play.")

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

    def predict_value(self, state: BoardState) -> float:
        if not self.model_loaded: return 0.0
        state_tensor = self._state_to_tensor(state)
        with torch.no_grad(): value = self.value_net(state_tensor).item()
        return value

    def get_best_move(self, state: BoardState) -> Optional[Move]:
        legal_moves = self.board.get_legal_moves(state)
        if not legal_moves: return None
        if not self.model_loaded: return random.choice(legal_moves)
        move_values = {move: self.predict_value(self.board.make_move(state, move)) for move in legal_moves}
        if not move_values: return random.choice(legal_moves)
        current_player = state.current_player
        if current_player == Player.WHITE: best_move = max(move_values, key=move_values.get)
        else: best_move = min(move_values, key=move_values.get)
        return best_move

# --- END OF FILE engines/dqn_agent.py ---