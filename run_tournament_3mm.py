# --- START OF FILE run_tournament.py ---
import time
import os
import sys
from datetime import datetime
from typing import Type, Optional, List  # Added List for type hinting
import random  # <--- ENSURE THIS IMPORT IS PRESENT
import pandas as pd  # Add this import at the top of the file

# Add project root to sys.path to allow imports from game and engines
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from game.board.ThreeMensMorrisBoard import ThreeMensMorrisBoard  # Specific for this phase
from game.board.Board import Board
from game.board.BoardState import BoardState
from game.util.Player import Player
from engines.Minimax import Minimax
from engines.MonteCarlo import MonteCarloTreeSearch
# from engines.MonteCarlo import MonteCarloTreeSearch # Uncomment if you add MCTS to tournament
from engines.Unbitable import GraphAgent  # "Perfect" agent for 3MM
from engines.StateValueAgent import StateValueAgent  # Trained on graph values
from engines.RL_DQN_Agent import RLDQNAgent  # Standard RL DQN
from engines.mDQN_Agent import MDQNAgent  # Modular DQN

# --- Tournament Configuration ---
BOARD_CLASS = ThreeMensMorrisBoard
NUM_GAMES_PER_MATCHUP = 20 # Reduced for quicker testing, increase later
MAX_MOVES_PER_GAME = 100

MODELS_DIR = "models"
STATES_GRAPH_DIR = "states_graph"


# --- Helper Function to Initialize Agents ---
def get_agent_instance(agent_name: str, board: Board, player_id: Player):
    """Creates an instance of the specified agent."""
    # print(f"Initializing agent: {agent_name} for Player {player_id.name}") # Can be verbose
    if agent_name == "Minimax-L1": return Minimax(board, max_depth=1)
    if agent_name == "Minimax-L2": return Minimax(board, max_depth=2)
    if agent_name == "Minimax-L3": return Minimax(board, max_depth=3)
    if agent_name == "GraphAgent":
        graph_path = os.path.join(STATES_GRAPH_DIR, "Threegame_graph_evaluated.txt")
        if not os.path.exists(graph_path): print(f"ERR: Graph file missing: {graph_path}"); return None
        return GraphAgent(board, graph_path)
    if agent_name == "StateValueAgent":
        model_path = os.path.join(MODELS_DIR, "3mm_state_value_model.pth")
        # if not os.path.exists(model_path): print(f"WARN: StateValueAgent model missing: {model_path}")
        return StateValueAgent(board, model_path=model_path)
    if agent_name == "RLDQNAgent":
        model_path = os.path.join(MODELS_DIR, f"{board.board_size}mm_rl_dqn_model.pth")
        # if not os.path.exists(model_path): print(f"WARN: RLDQNAgent model missing: {model_path}")
        return RLDQNAgent(board, model_path=model_path)
    if agent_name == "MDQNAgent":
        # print(f"WARN: Ensure MDQNAgent sub-models exist (e.g., models/3mm_mdqn_placement.pth)")
        return MDQNAgent(board)
    if agent_name == "Random":
        class RandomAgent:  # Define inside to keep 'random' module scope clean if it was the issue
            def __init__(self, board_ref: Board): self.board = board_ref

            def get_best_move(self, state: BoardState, *args, **kwargs):
                legal_moves = self.board.get_legal_moves(state)
                return random.choice(legal_moves) if legal_moves else None  # Uses the imported 'random' module

        return RandomAgent(board)
    if agent_name == "MCTS-Fast":
        agent = MonteCarloTreeSearch(board)
        agent.time_limit = 0.02  # Store time limit as an attribute
        return agent
    if agent_name == "MCTS-Deep":
        agent = MonteCarloTreeSearch(board)
        agent.time_limit = 0.1  # Store time limit as an attribute
        return agent
    print(f"ERROR: Unknown agent name: {agent_name}");
    return None


# Modify play_game function to track move times
def play_game(board: Board, agent1, agent2, max_moves=MAX_MOVES_PER_GAME):
    state = board.get_initial_board_state()
    game_moves = 0
    # Add timing tracking
    agent1_times = []
    agent2_times = []
    
    while not board.check_if_game_is_over(state) and game_moves < max_moves:
        current_agent = agent1 if state.current_player == Player.WHITE else agent2
        move = None
        try:
            # Measure move calculation time
            start_time = time.time()
            
            if isinstance(current_agent, Minimax):
                move = current_agent.get_best_move(state, max_time_seconds=0.5)
            elif isinstance(current_agent, MonteCarloTreeSearch):
                move = current_agent.get_best_move(state, max_time_seconds=getattr(current_agent, "time_limit", 0.1))
            else:
                move = current_agent.get_best_move(state)
                
            end_time = time.time()
            # Track time for appropriate agent
            if current_agent == agent1:
                agent1_times.append(end_time - start_time)
            else:
                agent2_times.append(end_time - start_time)
                
        except Exception as e:
            print(f"Error during {type(current_agent).__name__} get_best_move: {e}")
            legal_moves_left = board.get_legal_moves(state)
            if not legal_moves_left:
                break
            else:
                move = random.choice(legal_moves_left)
        if move is None: break
        state = board.make_move(state, move)
        game_moves += 1
        
    winner = board.get_winner(state)
    # Return timing information along with winner
    return winner, agent1_times, agent2_times


# --- Main Tournament Logic ---
def run_tournament(agents_to_test: List[str], num_games: int):
    board = BOARD_CLASS()
    results = {name: {"wins": 0, "losses": 0, "draws": 0, "played_white": 0, "played_black": 0} for name in
               agents_to_test}
    # Add timing stats
    timing_stats = {name: {"total_time": 0.0, "move_count": 0} for name in agents_to_test}
    
    # Initialize matchup matrix for Excel output
    matchup_results = {agent1: {agent2: 0 for agent2 in agents_to_test} for agent1 in agents_to_test}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"tournament_results_{board.board_size}mm_{timestamp}.txt"
    excel_filename = f"tournament_results_{board.board_size}mm_{timestamp}.xlsx"

    with open(results_filename, "w") as f_out:
        log_and_print = lambda msg: (f_out.write(msg + "\n"), print(msg))
        log_and_print(f"Tournament Started: {timestamp}")
        log_and_print(f"Board: {BOARD_CLASS.__name__}, Games per matchup: {num_games}")
        log_and_print(f"Agents: {', '.join(agents_to_test)}\n")

        for i in range(len(agents_to_test)):
            for j in range(i, len(agents_to_test)):
                agent1_name = agents_to_test[i]
                agent2_name = agents_to_test[j]
                if agent1_name == agent2_name and len(agents_to_test) > 1: continue
                log_and_print(f"--- Matchup: {agent1_name} vs {agent2_name} ---")
                matchup_wins_agent1_as_white = 0
                matchup_wins_agent2_as_white = 0
                matchup_draws = 0

                # Agent1 as White
                num_games_half1 = num_games // 2
                if num_games_half1 > 0:
                    log_and_print(f"  Playing {num_games_half1} games: {agent1_name} (W) vs {agent2_name} (B)...")
                    current_agent1 = get_agent_instance(agent1_name, board, Player.WHITE)
                    current_agent2 = get_agent_instance(agent2_name, board, Player.BLACK)
                    if not current_agent1 or not current_agent2: log_and_print(
                        "    Skipping (agent init error)"); continue
                    for game_num in range(num_games_half1):
                        winner, agent1_times, agent2_times = play_game(board, current_agent1, current_agent2)
                        if winner == Player.WHITE:
                            results[agent1_name]["wins"] += 1
                            results[agent2_name]["losses"] += 1
                            matchup_wins_agent1_as_white += 1
                            matchup_results[agent1_name][agent2_name] += 1  # Track win in matrix
                        elif winner == Player.BLACK:
                            results[agent2_name]["wins"] += 1
                            results[agent1_name]["losses"] += 1
                            matchup_results[agent2_name][agent1_name] += 1  # Track win in matrix
                        else:
                            results[agent1_name]["draws"] += 1
                            results[agent2_name]["draws"] += 1
                            matchup_draws += 1
                        results[agent1_name]["played_white"] += 1
                        results[agent2_name]["played_black"] += 1
                        if (game_num + 1) % (max(1, num_games_half1 // 5)) == 0: print(
                            f"    Game {game_num + 1}/{num_games_half1} done.")
                        
                        # After recording win/loss, add timing data
                        if agent1_times:
                            timing_stats[agent1_name]["total_time"] += sum(agent1_times)
                            timing_stats[agent1_name]["move_count"] += len(agent1_times)
                        if agent2_times:
                            timing_stats[agent2_name]["total_time"] += sum(agent2_times)
                            timing_stats[agent2_name]["move_count"] += len(agent2_times)

                # Agent2 as White (if different agents)
                num_games_half2 = num_games - num_games_half1
                if agent1_name != agent2_name and num_games_half2 > 0:
                    log_and_print(f"  Playing {num_games_half2} games: {agent2_name} (W) vs {agent1_name} (B)...")
                    current_agent1_sw = get_agent_instance(agent1_name, board, Player.BLACK)
                    current_agent2_sw = get_agent_instance(agent2_name, board, Player.WHITE)
                    if not current_agent1_sw or not current_agent2_sw: log_and_print(
                        "    Skipping swapped (agent init error)"); continue
                    for game_num in range(num_games_half2):
                        winner, agent1_times, agent2_times = play_game(board, current_agent2_sw, current_agent1_sw)
                        if winner == Player.WHITE:
                            results[agent2_name]["wins"] += 1
                            results[agent1_name]["losses"] += 1
                            matchup_wins_agent2_as_white += 1
                            matchup_results[agent2_name][agent1_name] += 1  # Track win in matrix
                        elif winner == Player.BLACK:
                            results[agent1_name]["wins"] += 1
                            results[agent2_name]["losses"] += 1
                            matchup_results[agent1_name][agent2_name] += 1  # Track win in matrix
                        else:
                            results[agent1_name]["draws"] += 1
                            results[agent2_name]["draws"] += 1
                            matchup_draws += 1
                        results[agent2_name]["played_white"] += 1
                        results[agent1_name]["played_black"] += 1
                        if (game_num + 1) % (max(1, num_games_half2 // 5)) == 0: print(
                            f"    Game {game_num + 1}/{num_games_half2} done.")
                        
                        # After updating matchup results, add timing data for swapped game
                        if agent1_times:
                            timing_stats[agent1_name]["total_time"] += sum(agent1_times)
                            timing_stats[agent1_name]["move_count"] += len(agent1_times)
                        if agent2_times:
                            timing_stats[agent2_name]["total_time"] += sum(agent2_times)
                            timing_stats[agent2_name]["move_count"] += len(agent2_times)

                log_and_print(
                    f"  Matchup Totals: {agent1_name} wins (as W): {matchup_wins_agent1_as_white}, {agent2_name} wins (as W): {matchup_wins_agent2_as_white}, Draws: {matchup_draws}\n")

        # Print traditional summary to terminal and text file
        log_and_print("\n--- Overall Tournament Results ---")
        for agent_name, stats in results.items():
            total_played = stats["wins"] + stats["losses"] + stats["draws"]
            win_rate = (stats["wins"] / total_played * 100) if total_played > 0 else 0
            log_and_print(f"Agent: {agent_name}")
            log_and_print(f"  Wins: {stats['wins']}, Losses: {stats['losses']}, Draws: {stats['draws']}")
            log_and_print(f"  Win Rate: {win_rate:.2f}%")
            log_and_print(f"  Played as White: {stats['played_white']}, Played as Black: {stats['played_black']}")
            log_and_print("--------------------")
        log_and_print(f"Tournament results saved to {results_filename}")
        
        # Create Excel file from the matrix data
        df = pd.DataFrame(matchup_results)
        
        # Add headers to explain the matrix
        with pd.ExcelWriter(excel_filename) as writer:
            df.to_excel(writer, sheet_name="Tournament Results")
            # Get the worksheet
            worksheet = writer.sheets["Tournament Results"]
            # Write header explanation
            worksheet.cell(row=1, column=1, value="Wins Matrix: Row player wins against Column player (Column starts as White)")
            
            # Add timing data to Excel output
            timing_df = pd.DataFrame({
                "Agent": list(timing_stats.keys()),
                "Avg Time (s)": [stats["total_time"] / max(1, stats["move_count"]) 
                                for stats in timing_stats.values()],
                "Total Moves": [stats["move_count"] for stats in timing_stats.values()]
            })
            timing_df.to_excel(writer, sheet_name="Timing Statistics")
        
        log_and_print(f"Tournament results matrix saved to {excel_filename}")
        
        # In the results section, add timing information
        log_and_print("\n--- Agent Timing Statistics ---")
        for agent_name, stats in timing_stats.items():
            if stats["move_count"] > 0:
                avg_time = stats["total_time"] / stats["move_count"]
                log_and_print(f"Agent: {agent_name}")
                log_and_print(f"  Total time: {stats['total_time']:.3f} seconds")
                log_and_print(f"  Total moves: {stats['move_count']}")
                log_and_print(f"  Average time per move: {avg_time:.6f} seconds")
                log_and_print("--------------------")


if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR, exist_ok=True)
    agents_for_3mm_tournament = [
        "Random", "Minimax-L1", "Minimax-L2", "Minimax-L3",
        "GraphAgent", "StateValueAgent", "RLDQNAgent", "MDQNAgent",
        "MCTS-Fast", "MCTS-Deep"
    ]
    # agents_for_3mm_tournament = ["Random", "Minimax-L2"] # For quick test
    run_tournament(agents_for_3mm_tournament, NUM_GAMES_PER_MATCHUP)

# --- END OF FILE run_tournament.py ---
