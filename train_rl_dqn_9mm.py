# --- START OF FILE train_rl_dqn_9mm.py ---
import torch
import os
import random
import numpy as np
from collections import deque
from typing import Optional, List, Tuple, Deque, Dict

from game.board.NineMensMorrisBoard import NineMensMorrisBoard  # <<< CHANGED
from game.board.BoardState import BoardState
from game.util.Player import Player
from game.util.Move import Move, MoveType
from game.util.Position import Position
from engines.RL_DQN_Agent import RLDQNAgent  # Assuming ActionValueNetwork inside might need adjustment

# --- Configuration ---
BOARD_TYPE = NineMensMorrisBoard  # <<< CHANGED
MODEL_SAVE_DIR = "models"
MODEL_SAVE_NAME = f"{BOARD_TYPE().board_size}mm_rl_dqn_model.pth"  # e.g., 24mm_rl_dqn_model.pth
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME)
LOG_FILE_PATH = f"training_rl_log_{BOARD_TYPE().board_size}mm.txt"

# Training Hyperparameters - ADJUST THESE SIGNIFICANTLY FOR 9MM
NUM_EPISODES = 200000  # <<< INCREASED DRAMATICALLY
MAX_STEPS_PER_EPISODE = 400  # <<< INCREASED
SAVE_MODEL_EVERY_N_EPISODES = 5000  # Save less frequently
PRINT_STATS_EVERY_N_EPISODES = 500  # Log less frequently

# Reward structure (Likely needs more shaping for 9MM)
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
REWARD_DRAW = 0.0
REWARD_MOVE = -0.005  # Smaller penalty per move
REWARD_MILL = 0.1  # Example: reward for forming a mill
REWARD_REMOVE = 0.2  # Example: reward for removing opponent piece
REWARD_LOSE_PIECE = -0.1  # Example: penalty for losing own piece

# RLDQNAgent Hyperparameters - ADJUST THESE FOR 9MM
RL_AGENT_HYPERPARAMS = {
    'learning_rate': 0.0001,  # Keep or slightly lower
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 500000,  # Much slower decay
    'target_update_freq': 1000,  # Update target net much less frequently
    'replay_memory_capacity': 500000,  # Much larger memory needed
    'batch_size': 256  # Increased batch size
}


def log_message(message): print(message);_ = open(LOG_FILE_PATH, "a").write(message + "\n")


# Helper to calculate reward (example with shaping)
def calculate_reward(state: BoardState, next_state: BoardState, done: bool, winner: Player):
    reward = REWARD_MOVE
    current_player = state.current_player
    opponent = current_player.opponent()

    # Check for mill formation by current player leading to removal state
    if not state.need_to_remove_piece and next_state.need_to_remove_piece:
        reward += REWARD_MILL

    # Check if opponent lost a piece (reward for successful removal)
    # We check the difference in pieces *after* a potential removal move by current player
    # This assumes the current player is the one causing the piece count change for the opponent.
    if state.pieces_from_player_currently_on_board[opponent] > next_state.pieces_from_player_currently_on_board[
        opponent]:
        reward += REWARD_REMOVE

    # Check if current player lost a piece (penalty) - happens on opponent's turn result
    # This part is tricky in self-play, as reward is assigned to the action *leading* to the state.
    # A simpler approach focuses on terminal rewards and immediate events like mills/removals.
    # if state.pieces_from_player_currently_on_board[current_player] > next_state.pieces_from_player_currently_on_board[current_player]:
    #      reward += REWARD_LOSE_PIECE # This might be double counted if it leads to loss

    if done:
        if winner == Player.NONE:
            reward += REWARD_DRAW
        elif winner == current_player:
            reward += REWARD_WIN
        elif winner == opponent:
            reward += REWARD_LOSE
        # Clear the move penalty if the game ended? Optional.
        # reward -= REWARD_MOVE

    return reward


def train_rl_agent_9mm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")

    board_instance = BOARD_TYPE()
    agent = RLDQNAgent(board_instance, model_path=MODEL_SAVE_PATH, device=device, **RL_AGENT_HYPERPARAMS)

    rewards_window = deque(maxlen=PRINT_STATS_EVERY_N_EPISODES)
    durations_window = deque(maxlen=PRINT_STATS_EVERY_N_EPISODES)
    white_wins = deque(maxlen=PRINT_STATS_EVERY_N_EPISODES)

    log_message(f"RL DQN training for 9MM: {NUM_EPISODES} eps. Model: {MODEL_SAVE_PATH}. Actions: {agent.num_actions}")

    for i_episode in range(1, NUM_EPISODES + 1):
        state = board_instance.get_initial_board_state();
        episode_reward = 0.0
        for t in range(MAX_STEPS_PER_EPISODE):
            player = state.current_player
            move, action_idx = agent.select_action_epsilon_greedy(state);
            agent.steps_done += 1

            if move is None or action_idx is None:
                done = True;
                winner = board_instance.get_winner(state)
                # Calculate reward based on the terminal state for the player whose turn it was
                step_reward = REWARD_DRAW if winner == Player.NONE else (
                    REWARD_WIN if winner == player else REWARD_LOSE)
                next_state_tensor = None
                break  # End episode immediately

            next_state = board_instance.make_move(state, move)
            done = board_instance.check_if_game_is_over(next_state)
            winner = board_instance.get_winner(next_state) if done else Player.NONE

            # Calculate reward for the (state, action -> next_state) transition
            step_reward = calculate_reward(state, next_state, done, winner)
            episode_reward += step_reward

            st_tensor = agent._state_to_tensor(state)
            nst_tensor = agent._state_to_tensor(next_state) if not done else None
            agent.memory.push((st_tensor, action_idx, step_reward, nst_tensor, done));
            state = next_state

            if len(agent.memory) > agent.batch_size * 10:  # Start training only after buffer is reasonably full
                for _ in range(4):  # Perform multiple optimization steps per game step if needed
                    agent.optimize_model()

            if done:
                if winner == Player.WHITE:
                    white_wins.append(1.0)
                elif winner == Player.BLACK:
                    white_wins.append(0.0)
                else:
                    white_wins.append(0.5)
                break

        rewards_window.append(episode_reward);
        durations_window.append(t + 1)
        agent.update_target_network_if_needed()  # Updates based on episode count

        if i_episode % PRINT_STATS_EVERY_N_EPISODES == 0:
            avg_r = np.mean(rewards_window) if rewards_window else 0
            avg_d = np.mean(durations_window) if durations_window else 0
            wr_w = sum(w == 1.0 for w in white_wins) / len(white_wins) * 100 if white_wins else 0
            dr = sum(w == 0.5 for w in white_wins) / len(white_wins) * 100 if white_wins else 0
            eps = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * np.exp(
                -1. * agent.steps_done / agent.epsilon_decay)
            log_message(
                f"Ep {i_episode}|AvgRew:{avg_r:.2f}|AvgSteps:{avg_d:.1f}|Eps:{eps:.3f}|W%:{wr_w:.1f}|D%:{dr:.1f}")
            if len(white_wins) >= PRINT_STATS_EVERY_N_EPISODES: white_wins.clear()  # Clear after logging

        if i_episode % SAVE_MODEL_EVERY_N_EPISODES == 0:
            agent.save_model(MODEL_SAVE_PATH)
    log_message("Training complete for 9MM RLDQN.");
    agent.save_model(MODEL_SAVE_PATH)


if __name__ == "__main__":
    if not os.path.exists(MODEL_SAVE_DIR): os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    if os.path.exists(LOG_FILE_PATH):
        try:
            os.remove(LOG_FILE_PATH)
        except OSError as e:
            print(f"Warning: Could not remove log: {e}")
    train_rl_agent_9mm()
# --- END OF FILE train_rl_dqn_9mm.py ---