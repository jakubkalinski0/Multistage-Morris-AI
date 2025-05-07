# --- START OF FILE train_rl_dqn.py ---
import torch
import os
import random
import numpy as np
from collections import deque
from typing import Optional, List, Tuple, Deque, Dict  # Added Dict

# Adjust imports as necessary if run from a different location or project structure
from game.board.ThreeMensMorrisBoard import ThreeMensMorrisBoard  # Or other board
from game.board.BoardState import BoardState
from game.util.Player import Player
from game.util.Move import Move, MoveType  # Import Move and MoveType
from game.util.Position import Position  # Import Position
from engines.RL_DQN_Agent import RLDQNAgent  # Import the RL DQN Agent

# --- Configuration ---
BOARD_TYPE = ThreeMensMorrisBoard  # Change for other boards
MODEL_SAVE_DIR = "models"
MODEL_SAVE_NAME = f"{BOARD_TYPE().board_size}mm_rl_dqn_model.pth"  # Dynamic model name
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME)
LOG_FILE_PATH = "training_rl_log.txt"  # Log file in the root directory

# Training Hyperparameters
NUM_EPISODES = 10000  # Total number of games to play for training
MAX_STEPS_PER_EPISODE = 150  # For 3MM, games are short. Adjust for larger boards.
SAVE_MODEL_EVERY_N_EPISODES = 500
PRINT_STATS_EVERY_N_EPISODES = 100

# Reward structure
REWARD_WIN = 10.0  # Increased win reward
REWARD_LOSE = -10.0  # Increased lose penalty
REWARD_DRAW = 0.0
REWARD_MOVE = -0.05  # Small penalty for each move to encourage faster wins


def log_message(message):
    print(message)
    with open(LOG_FILE_PATH, "a") as f:
        f.write(message + "\n")


def train_rl_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")

    board_instance = BOARD_TYPE()
    # Initialize agent (it might load a pre-existing model if path exists)
    agent = RLDQNAgent(board_instance, model_path=MODEL_SAVE_PATH, device=device)

    episode_rewards_window = deque(maxlen=PRINT_STATS_EVERY_N_EPISODES)
    episode_durations_window = deque(maxlen=PRINT_STATS_EVERY_N_EPISODES)
    wins_for_player1 = 0  # Assuming agent is player 1 in self-play initially

    log_message(f"Starting RL DQN training for {NUM_EPISODES} episodes...")
    log_message(f"Model will be saved to: {MODEL_SAVE_PATH}")
    log_message(f"Action space size: {agent.num_actions}")

    for i_episode in range(1, NUM_EPISODES + 1):
        state = board_instance.get_initial_board_state()
        episode_reward = 0

        for t in range(MAX_STEPS_PER_EPISODE):
            current_player_obj = state.current_player  # Player whose turn it is

            # Agent selects action
            move_obj, action_idx = agent.select_action_epsilon_greedy(state)

            if move_obj is None or action_idx is None:
                # This implies the current player loses or it's a draw if no moves
                done = True
                # Determine reward: if current player has no moves, they lose (unless it's a draw already)
                # The board.get_winner() should reflect this.
                winner = board_instance.get_winner(state)  # Check winner based on current state
                if winner == Player.NONE:
                    reward = REWARD_DRAW
                elif winner == current_player_obj:
                    reward = REWARD_WIN  # Should not happen often if no moves
                else:
                    reward = REWARD_LOSE
                next_state_tensor = None  # Terminal state
                # print(f"No move/action for player {current_player_obj.name}. Done. Reward: {reward}")
                break  # End episode

            # Execute move and get next state
            next_state = board_instance.make_move(state, move_obj)

            reward = REWARD_MOVE  # Base reward for taking a step

            done = board_instance.check_if_game_is_over(next_state)
            if done:
                winner = board_instance.get_winner(next_state)
                if winner == Player.NONE:
                    reward += REWARD_DRAW
                # If current_player_obj made the winning move
                elif winner == current_player_obj:
                    reward += REWARD_WIN
                    if current_player_obj == Player.WHITE: wins_for_player1 += 1  # If agent was WHITE
                # If current_player_obj made a move that led to opponent winning
                elif winner == current_player_obj.opponent():
                    reward += REWARD_LOSE
                else:  # Should not happen if winner is NONE, WHITE or BLACK
                    reward += REWARD_DRAW

            episode_reward += reward  # Accumulate step rewards if any (currently only terminal + move penalty)

            # Store experience in replay memory
            state_tensor = agent._state_to_tensor(state)
            next_state_tensor = agent._state_to_tensor(next_state) if not done else None

            agent.memory.push((state_tensor, action_idx, reward, next_state_tensor, done))

            state = next_state  # Move to the next state

            if len(agent.memory) > agent.batch_size:
                agent.optimize_model()

            if done:
                break

        episode_rewards_window.append(episode_reward)
        episode_durations_window.append(t + 1)

        # Update the target network periodically (the method handles the frequency check)
        agent.update_target_network_if_needed()  # <--- CORRECTED METHOD CALL

        if i_episode % PRINT_STATS_EVERY_N_EPISODES == 0:
            avg_reward = np.mean(episode_rewards_window) if episode_rewards_window else 0
            avg_duration = np.mean(episode_durations_window) if episode_durations_window else 0
            current_epsilon = agent.epsilon_end + (agent.epsilon_start - agent.epsilon_end) * \
                              np.exp(-1. * agent.steps_done / agent.epsilon_decay)
            log_message(
                f"Ep {i_episode}/{NUM_EPISODES} | Avg Rew (last {PRINT_STATS_EVERY_N_EPISODES}): {avg_reward:.2f} | Avg Steps: {avg_duration:.1f} | Epsilon: {current_epsilon:.3f} | P1 Wins (last {PRINT_STATS_EVERY_N_EPISODES}): {wins_for_player1}")
            wins_for_player1 = 0  # Reset win counter for next block

        if i_episode % SAVE_MODEL_EVERY_N_EPISODES == 0:
            agent.save_model(MODEL_SAVE_PATH)
            log_message(f"Model saved at episode {i_episode}")

    log_message("Training complete.")
    agent.save_model(MODEL_SAVE_PATH)  # Final save


if __name__ == "__main__":
    # Ensure the models directory exists
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)
    train_rl_agent()

# --- END OF FILE train_rl_dqn.py ---