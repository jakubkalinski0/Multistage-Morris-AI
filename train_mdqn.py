# --- START OF FILE train_mdqn.py ---
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
from engines.mDQN_Agent import MDQNAgent  # Import the main MDQN dispatcher
from game.GamePhase import GamePhase  # Make sure GamePhase is imported

# --- Configuration ---
BOARD_TYPE = ThreeMensMorrisBoard
# MODEL_SAVE_DIR will be handled by the agent for its sub-models, but we define a root for models
ROOT_MODELS_DIR = "models"  # Used to ensure the base 'models' directory exists if needed
LOG_FILE_PATH = f"training_mdqn_log_{BOARD_TYPE().board_size}mm.txt"  # Log file in the root directory

# Training Hyperparameters
NUM_EPISODES = 10000
MAX_STEPS_PER_EPISODE = 100
# SAVE_MODEL_EVERY_N_EPISODES: MDQNAgent.save_models() will be called based on this
SAVE_MODEL_EVERY_N_EPISODES = 1000
PRINT_STATS_EVERY_N_EPISODES = 100

MDQN_HYPERPARAMS = {  # Passed to MDQNAgent, which passes to sub-agents
    'learning_rate': 0.00025,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.05,
    'epsilon_decay': 50000,  # Slower decay for more exploration steps
    'target_update_freq': 20,  # Each sub-agent's target net updates based on this many global episodes
    'replay_memory_capacity': 30000,
    'batch_size': 128
}

# Reward structure
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
REWARD_DRAW = 0.0
REWARD_MOVE = -0.01  # Small penalty for each move to encourage faster wins


def log_message(message):
    print(message)
    with open(LOG_FILE_PATH, "a") as f:
        f.write(message + "\n")


def train_mdqn_agent():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_message(f"Using device: {device}")

    board_instance = BOARD_TYPE()
    # MDQNAgent will initialize its sub-agents.
    # The model_path_prefix is used by MDQNAgent to construct paths for its sub-agents.
    agent = MDQNAgent(board_instance, device=device, **MDQN_HYPERPARAMS)

    episode_rewards_window = deque(maxlen=PRINT_STATS_EVERY_N_EPISODES)
    episode_durations_window = deque(maxlen=PRINT_STATS_EVERY_N_EPISODES)
    white_wins_window = deque(maxlen=PRINT_STATS_EVERY_N_EPISODES)  # Tracks outcome for White

    log_message(f"Starting mDQN training for {NUM_EPISODES} episodes...")
    # Model save paths are handled internally by the sub-agents within MDQNAgent

    for i_episode in range(1, NUM_EPISODES + 1):
        state = board_instance.get_initial_board_state()
        episode_reward_sum = 0.0

        for t in range(MAX_STEPS_PER_EPISODE):
            current_player_obj = state.current_player
            # Determine which sub-agent is responsible for the decision in the current state
            sub_agent_key_for_decision, active_sub_agent = agent._get_current_sub_agent_key_and_instance(state)

            move_obj, action_idx = agent.select_action_epsilon_greedy(state)
            # Crucial: Epsilon decay step counter is managed by the main MDQNAgent
            agent.steps_done += 1

            if move_obj is None or action_idx is None:
                done = True
                winner = board_instance.get_winner(state)
                if winner == Player.NONE:
                    step_reward = REWARD_DRAW
                elif winner == current_player_obj:
                    step_reward = REWARD_WIN
                else:
                    step_reward = REWARD_LOSE
                next_state_tensor = None
                # print(f"No move/action for player {current_player_obj.name}. Done. Reward: {step_reward}")
                break

            next_state = board_instance.make_move(state, move_obj)
            step_reward = REWARD_MOVE

            done = board_instance.check_if_game_is_over(next_state)
            if done:
                winner = board_instance.get_winner(next_state)
                if winner == Player.NONE:
                    step_reward += REWARD_DRAW
                elif winner == current_player_obj:
                    step_reward += REWARD_WIN
                elif winner == current_player_obj.opponent():
                    step_reward += REWARD_LOSE

            episode_reward_sum += step_reward

            # Use the active_sub_agent's state_to_tensor for consistency
            state_tensor = active_sub_agent._state_to_tensor(state)
            next_state_tensor = active_sub_agent._state_to_tensor(next_state) if not done else None

            # Push experience to the memory of the sub-agent that made the decision
            # Pass 'state' to determine which sub-agent's memory to use
            agent.push_experience(state_tensor, action_idx, step_reward, next_state_tensor, done, state)

            state = next_state

            # Optimize the sub-agent that was responsible for the decision, if its buffer is full
            if len(active_sub_agent.memory) > active_sub_agent.batch_size:
                agent.optimize_sub_agent(sub_agent_key_for_decision)

            if done:
                if winner == Player.WHITE:
                    white_wins_window.append(1.0)
                elif winner == Player.BLACK:
                    white_wins_window.append(0.0)  # White loses
                else:
                    white_wins_window.append(0.5)  # Draw
                break

        episode_rewards_window.append(episode_reward_sum)
        episode_durations_window.append(t + 1)

        # This method in MDQNAgent will increment its internal episode counter
        # and call update_target_network_sub on each sub-agent if their frequency matches.
        agent.update_target_networks_globally()

        if i_episode % PRINT_STATS_EVERY_N_EPISODES == 0:
            avg_reward = np.mean(episode_rewards_window) if episode_rewards_window else 0
            avg_duration = np.mean(episode_durations_window) if episode_durations_window else 0

            # Calculate win rate for White from the window
            num_white_wins = sum(1 for outcome in white_wins_window if outcome == 1.0)
            num_draws = sum(1 for outcome in white_wins_window if outcome == 0.5)
            total_games_in_window = len(white_wins_window)

            win_rate_white_percent = (num_white_wins / total_games_in_window * 100) if total_games_in_window > 0 else 0
            draw_rate_percent = (num_draws / total_games_in_window * 100) if total_games_in_window > 0 else 0

            # Use epsilon from one of the sub-agents (they should share steps_done or have similar decay)
            # Here, we use the epsilon parameters defined for the sub-agents via MDQN_HYPERPARAMS
            current_epsilon_main = MDQN_HYPERPARAMS['epsilon_end'] + \
                                   (MDQN_HYPERPARAMS['epsilon_start'] - MDQN_HYPERPARAMS['epsilon_end']) * \
                                   np.exp(-1. * agent.steps_done / MDQN_HYPERPARAMS['epsilon_decay'])
            log_message(
                f"Ep {i_episode}/{NUM_EPISODES} | Avg Rew: {avg_reward:.2f} | Avg Steps: {avg_duration:.1f} | Epsilon: {current_epsilon_main:.3f} | White Win %: {win_rate_white_percent:.1f}% | Draw %: {draw_rate_percent:.1f}%")
            # Clear window after logging for fresh stats in next block
            if total_games_in_window == PRINT_STATS_EVERY_N_EPISODES:
                white_wins_window.clear()  # Or re-initialize if needed for exact window size

        if i_episode % SAVE_MODEL_EVERY_N_EPISODES == 0:
            agent.save_models()  # Saves all sub-agent models
            log_message(f"All mDQN sub-agent models saved at episode {i_episode}")

    log_message("mDQN Training complete.")
    agent.save_models()


if __name__ == "__main__":
    # Ensure the main 'models' directory exists; sub-agents might create sub-folders or save here.
    if not os.path.exists(ROOT_MODELS_DIR):
        os.makedirs(ROOT_MODELS_DIR, exist_ok=True)

    if os.path.exists(LOG_FILE_PATH):
        try:
            os.remove(LOG_FILE_PATH)  # Clear log file at start
        except OSError as e:
            print(f"Warning: Could not remove old log file {LOG_FILE_PATH}: {e}")
    train_mdqn_agent()

# --- END OF FILE train_mdqn.py ---