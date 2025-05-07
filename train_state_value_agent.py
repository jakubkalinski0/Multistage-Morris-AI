# --- START OF FILE train_state_value_agent.py ---

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import networkx as nx
from typing import Optional, List, Tuple

# Adjust imports based on your project structure (assuming run from root)
from game.board.ThreeMensMorrisBoard import ThreeMensMorrisBoard
from game.board.Board import Board
from game.board.BoardState import BoardState
from game.util.Player import Player
# Updated import to reflect new class names
from engines.StateValueAgent import StateValueAgent, ValueNetwork

# --- Configuration ---
BOARD_TYPE = ThreeMensMorrisBoard
EVALUATED_GRAPH_PATH = "states_graph/Threegame_graph_evaluated.txt"
MODEL_SAVE_DIR = "models"
MODEL_SAVE_NAME = "3mm_state_value_model.pth"  # Updated model name
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME)

INPUT_SIZE = StateValueAgent.DEFAULT_INPUT_SIZE_3MM  # Use default from agent
OUTPUT_SIZE = 1  # Predicting a single state value

# Training Hyperparameters
EPOCHS = 50  # Number of training epochs (adjust as needed)
BATCH_SIZE = 512  # Increased batch size for potentially large dataset
LEARNING_RATE = 0.001
TEST_SPLIT = 0.15  # Use 15% of data for testing
RANDOM_SEED = 42  # For reproducibility


# --- Helper Functions ---
def state_to_feature_list(state: BoardState, board_size: int) -> List[float]:
    """Converts a BoardState object into a list of features for the dataset."""
    # Replicated from StateValueAgent for data generation consistency
    board_repr = []
    for i in range(board_size):
        player = state.get_player_at_position(i)
        if player == Player.NONE:
            board_repr.extend([1.0, 0.0, 0.0])
        elif player == Player.WHITE:
            board_repr.extend([0.0, 1.0, 0.0])
        else:  # Player.BLACK
            board_repr.extend([0.0, 0.0, 1.0])

    current_player_repr = [1.0] if state.current_player == Player.WHITE else [-1.0]
    pieces_left_w = float(state.pieces_left_to_place_by_player[Player.WHITE])
    pieces_left_b = float(state.pieces_left_to_place_by_player[Player.BLACK])
    pieces_left_repr = [pieces_left_w, pieces_left_b]
    pieces_on_w = float(state.pieces_from_player_currently_on_board[Player.WHITE])
    pieces_on_b = float(state.pieces_from_player_currently_on_board[Player.BLACK])
    pieces_on_repr = [pieces_on_w, pieces_on_b]
    remove_repr = [1.0] if state.need_to_remove_piece else [0.0]

    features = board_repr + current_player_repr + pieces_left_repr + pieces_on_repr + remove_repr

    # Dynamic calculation for expected size based on board_size
    # INPUT_SIZE here is StateValueAgent.DEFAULT_INPUT_SIZE_3MM, which is for board_size 9
    expected_size_for_this_board = board_size * 3 + 1 + 2 + 2 + 1
    if len(features) != expected_size_for_this_board:
        raise ValueError(
            f"Feature length mismatch: expected {expected_size_for_this_board}, got {len(features)} for board size {board_size}")
    return features


# --- Dataset Class ---
class MorrisValueDataset(Dataset):
    """PyTorch Dataset for Morris state value prediction."""

    def __init__(self, data, device):
        self.features = torch.tensor([item[0] for item in data], dtype=torch.float32).to(device)
        self.labels = torch.tensor([[item[1]] for item in data], dtype=torch.float32).to(device)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# --- Data Generation ---
def generate_training_data(graph_path: str, board: Board) -> List[Tuple[List[float], float]]:
    """Generates (features, value) tuples from the evaluated graph."""
    print(f"Generating training data from graph: {graph_path}")
    if not os.path.exists(graph_path):
        print(f"ERROR: Evaluated graph file not found at {graph_path}")
        return []

    try:
        graph = nx.read_adjlist(graph_path, create_using=nx.DiGraph())
        print(f"Graph loaded successfully with {graph.number_of_nodes()} nodes.")
    except Exception as e:
        print(f"Error reading graph file: {e}")
        return []

    training_data = []
    processed_nodes = 0
    skipped_nodes = 0
    board_size = board.board_size  # Should be 9 for 3MM

    # Create a single BoardState object to reuse or re-init in loop
    temp_board_state = BoardState(board_size, board.pieces_per_player)

    for node_str in graph.nodes():
        try:
            if node_str.endswith("1"):
                target_value = 1.0; state_int_str = node_str[:-1]
            elif node_str.endswith("2"):
                target_value = -1.0; state_int_str = node_str[:-1]
            else:
                target_value = 0.0; state_int_str = node_str[:-1] if node_str.endswith("0") else node_str

            if not state_int_str.isdigit(): skipped_nodes += 1; continue
            state_int = int(state_int_str)

            temp_board_state = BoardState(board_size, board.pieces_per_player)  # Re-init for safety
            temp_board_state.from_int(state_int)

            features = state_to_feature_list(temp_board_state, board_size)
            training_data.append((features, target_value))
            processed_nodes += 1
        except ValueError as e:
            skipped_nodes += 1
        except Exception as e:
            skipped_nodes += 1
        if (processed_nodes + skipped_nodes) % 20000 == 0:
            print(f"  Processed: {processed_nodes}, Skipped: {skipped_nodes}")

    print(f"Data generation complete. Processed nodes: {processed_nodes}, Skipped: {skipped_nodes}")
    print(f"Total training samples: {len(training_data)}")
    if not training_data: print("Warning: No training data could be generated.")
    return training_data


# --- Training Loop ---
def train_model(data: List[Tuple[List[float], float]], model_save_path: str, device: torch.device):
    """Trains the ValueNetwork model using the generated data."""
    if not data: print("No training data provided. Exiting training."); return

    random.seed(RANDOM_SEED);
    np.random.seed(RANDOM_SEED);
    torch.manual_seed(RANDOM_SEED)
    if device == torch.device("cuda"): torch.cuda.manual_seed_all(RANDOM_SEED)

    random.shuffle(data)
    split_idx = int(len(data) * (1 - TEST_SPLIT))
    train_data, test_data = data[:split_idx], data[split_idx:]
    if not train_data or not test_data: print("Error: Not enough data for train/test split."); return

    train_dataset = MorrisValueDataset(train_data, device)
    test_dataset = MorrisValueDataset(test_data, device)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Training set size: {len(train_dataset)}, Test set size: {len(test_dataset)}")

    model = ValueNetwork(INPUT_SIZE, OUTPUT_SIZE).to(device)  # Use ValueNetwork
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print("Starting training for StateValueAgent model...")
    for epoch in range(EPOCHS):
        model.train();
        running_loss = 0.0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward();
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)
        model.eval();
        test_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}")
    try:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        torch.save(model.state_dict(), model_save_path)
        print(f"StateValueAgent model training complete. Model saved to {model_save_path}")
    except Exception as e:
        print(f"Error saving model: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    board = BOARD_TYPE()
    if board.board_size != 9:  # This script is specifically for 3MM graph data
        print("ERROR: This training script is configured for Three Men's Morris (board size 9).")
        print("Please ensure BOARD_TYPE and EVALUATED_GRAPH_PATH are set for 3 Men's Morris.")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        training_data = generate_training_data(EVALUATED_GRAPH_PATH, board)
        if training_data:
            train_model(training_data, MODEL_SAVE_PATH, device)
        else:
            print("Training aborted due to lack of data.")
# --- END OF FILE train_state_value_agent.py ---