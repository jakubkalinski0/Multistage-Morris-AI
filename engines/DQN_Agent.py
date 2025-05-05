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

# Define the Neural Network structure (QNetwork class remains the same)
class QNetwork(nn.Module):
    """Simple MLP to predict state value."""
    def __init__(self, input_size, output_size=1):
        super(QNetwork, self).__init__()
        # Adjust layers/neurons as needed
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_size) # Output a single value

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # Tanh forces output between -1 and 1, matching our target values
        x = torch.tanh(self.layer3(x))
        return x

# DQNAgent class definition follows...
# (Reszta kodu DQNAgent pozostaje taka sama jak w poprzedniej odpowiedzi)
# ... (cała reszta klasy DQNAgent bez zmian) ...

class DQNAgent:
    """
    Agent using a trained network to evaluate board states and choose moves.
    The network is trained via supervised learning on state values derived
    from an evaluated game graph (like the one for 3 Men's Morris).
    """
    # Determine input size based on state representation
    # Default for 3MM: 9 pos * 3 states + 1 current_player + 2 pieces_to_place + 2 pieces_on_board + 1 needs_remove
    INPUT_SIZE = 9 * 3 + 1 + 2 + 2 + 1 # = 33

    def __init__(self, board: Board, model_path: Optional[str] = "models/dqn_3mm_value_model.pth", device: Optional[str] = None):
        """
        Initializes the DQN agent.

        Args:
            board (Board): The game board instance.
            model_path (Optional[str]): Path to the trained model weights.
            device (Optional[str]): Device to run the model on ('cuda' or 'cpu'). Detects automatically if None.
        """
        self.board = board
        self.board_size = board.board_size

        # Adjust input size dynamically if not 3MM, though the model is trained for 3MM.
        self.INPUT_SIZE = self.board_size * 3 + 1 + 2 + 2 + 1
        if self.board_size != 9:
             print(f"Warning: This DQN Agent model was trained for 3 Men's Morris (board size 9). Using it for board size {self.board_size}. Input size adjusted to {self.INPUT_SIZE}.")

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DQNAgent attempting to use device: {self.device}")

        # Initialize the policy network
        self.policy_net = QNetwork(self.INPUT_SIZE).to(self.device)
        self.model_loaded = False

        # Load the trained model if path is provided and exists
        if model_path:
            # Make sure the path is absolute or relative to the execution directory (root)
            if not os.path.isabs(model_path):
                 # Assuming train_dqn.py is in root, model_path should be relative to root
                 # If PyGame.py calls this, it adjusts the path relatively
                 pass # Use the path as is if relative, hoping it's correct from root
            if os.path.exists(model_path):
                try:
                    # Load the state dict, ensuring it maps to the correct device
                    self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
                    self.policy_net.eval() # Set the network to evaluation mode
                    self.model_loaded = True
                    print(f"Successfully loaded trained model from {model_path}")
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}. Agent will play randomly.")
            else:
                print(f"Model file not found at {model_path}. Agent will play randomly.")
        else:
             print("No model path provided. Agent will play randomly.")


    def _state_to_tensor(self, state: BoardState) -> torch.Tensor:
        """Converts a BoardState object into a PyTorch tensor feature vector."""
        # 1. Board representation (one-hot encoding per position)
        board_repr = []
        for i in range(self.board_size):
            player = state.get_player_at_position(i)
            if player == Player.NONE:
                board_repr.extend([1.0, 0.0, 0.0])
            elif player == Player.WHITE:
                board_repr.extend([0.0, 1.0, 0.0])
            else: # Player.BLACK
                board_repr.extend([0.0, 0.0, 1.0])

        # 2. Current player (-1 for Black, 1 for White)
        current_player_repr = [1.0] if state.current_player == Player.WHITE else [-1.0]

        # 3. Pieces left to place (raw count)
        pieces_left_w = float(state.pieces_left_to_place_by_player[Player.WHITE])
        pieces_left_b = float(state.pieces_left_to_place_by_player[Player.BLACK])
        pieces_left_repr = [pieces_left_w, pieces_left_b]

        # 4. Pieces currently on board
        pieces_on_w = float(state.pieces_from_player_currently_on_board[Player.WHITE])
        pieces_on_b = float(state.pieces_from_player_currently_on_board[Player.BLACK])
        pieces_on_repr = [pieces_on_w, pieces_on_b]

        # 5. Need to remove piece flag
        remove_repr = [1.0] if state.need_to_remove_piece else [0.0]

        # Combine all features
        features = board_repr + current_player_repr + pieces_left_repr + pieces_on_repr + remove_repr

        # Ensure length matches expected INPUT_SIZE
        if len(features) != self.INPUT_SIZE:
            raise ValueError(f"Feature length mismatch in _state_to_tensor: expected {self.INPUT_SIZE}, got {len(features)}")

        # Convert to tensor
        state_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0) # Add batch dimension
        return state_tensor.to(self.device)

    def predict_value(self, state: BoardState) -> float:
        """Predicts the value of a given state using the network."""
        if not self.model_loaded:
             return 0.0 # Return neutral value if model isn't loaded

        state_tensor = self._state_to_tensor(state)
        with torch.no_grad():
            # Network outputs a value between -1 (Black wins) and 1 (White wins)
            value = self.policy_net(state_tensor).item()
        return value

    def get_best_move(self, state: BoardState) -> Optional[Move]:
        """
        Selects the best move by evaluating the resulting states of legal moves.
        Uses the trained network to predict state values.
        If no model is loaded, plays randomly.
        """
        legal_moves = self.board.get_legal_moves(state)
        if not legal_moves:
            return None

        # If model wasn't loaded successfully, play randomly
        if not self.model_loaded:
            # print("DQN Agent (Model not loaded) playing randomly.")
            return random.choice(legal_moves)

        move_values = {}
        for move in legal_moves:
            # Simulate the move
            next_state = self.board.make_move(state, move)
            # Predict the value of the resulting state (from White's perspective)
            value = self.predict_value(next_state)
            move_values[move] = value

        current_player = state.current_player
        if not move_values: # Should not happen if legal_moves is not empty
            print("Warning: No move values calculated despite legal moves existing.")
            return random.choice(legal_moves)

        if current_player == Player.WHITE:
            # White wants to maximize the value (move towards +1)
            best_move = max(move_values, key=move_values.get)
            # print(f"DQN Agent (WHITE) chose move leading to value: {move_values[best_move]:.3f}")
        else: # Player.BLACK
            # Black wants to minimize the value (move towards -1)
            best_move = min(move_values, key=move_values.get)
            # print(f"DQN Agent (BLACK) chose move leading to value: {move_values[best_move]:.3f}")

        return best_move