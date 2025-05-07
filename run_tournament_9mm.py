# --- START OF FILE run_tournament_9mm.py ---
import os
import sys
from datetime import datetime
from typing import Type, Optional, List
import random
import time  # Import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path: sys.path.insert(0, project_root)

from game.board.NineMensMorrisBoard import NineMensMorrisBoard  # <<< CHANGED
from game.board.Board import Board
from game.board.BoardState import BoardState
from game.util.Player import Player
from engines.Minimax import Minimax
from engines.RL_DQN_Agent import RLDQNAgent
from engines.mDQN_Agent import MDQNAgent

# from engines.MonteCarlo import MonteCarloTreeSearch # Add back if testing MCTS

# --- Tournament Configuration ---
BOARD_CLASS = NineMensMorrisBoard  # <<< CHANGED
NUM_GAMES_PER_MATCHUP = 20  # Start very low for 9MM due to game length/complexity
MAX_MOVES_PER_GAME = 350  # <<< INCREASED SIGNIFICANTLY

MODELS_DIR = "models"


# --- Helper Function to Initialize Agents (Adapted for 9MM) ---
def get_agent_instance(agent_name: str, board: Board, player_id: Player):
    print(f"Initializing agent for 9MM: {agent_name} for Player {player_id.name}")
    if agent_name == "Minimax-L1": return Minimax(board, max_depth=1)
    if agent_name == "Minimax-L2": return Minimax(board, max_depth=2)  # L2 is already challenging for 9MM
    # Higher Minimax levels are likely infeasible without heavy optimization/pruning

    if agent_name == "RLDQNAgent":
        model_path = os.path.join(MODELS_DIR, f"{board.board_size}mm_rl_dqn_model.pth")  # e.g., 24mm...
        # if not os.path.exists(model_path): print(f"WARN: RLDQNAgent 9MM model missing: {model_path}")
        return RLDQNAgent(board, model_path=model_path)
    if agent_name == "MDQNAgent":
        # print(f"WARN: Ensure MDQN sub-models for {board.board_size}MM exist.")
        return MDQNAgent(board)
    if agent_name == "Random":
        class RandomAgent:
            def __init__(self, board_ref: Board): self.board = board_ref

            def get_best_move(self, state: BoardState, *args, **kwargs):
                lm = self.board.get_legal_moves(state);
                return random.choice(lm) if lm else None

        return RandomAgent(board)
    # Add MCTS back here if needed
    # if agent_name == "MCTS-1s": return MonteCarloTreeSearch(board) # Default depth

    print(f"ERROR: Unknown agent name for 9MM tournament: {agent_name}");
    return None


# --- Game Playing Function (play_game - use the version from 6MM) ---
def play_game(board: Board, agent1, agent2, max_moves=MAX_MOVES_PER_GAME):
    state = board.get_initial_board_state();
    game_moves = 0
    while not board.check_if_game_is_over(state) and game_moves < max_moves:
        current_agent = agent1 if state.current_player == Player.WHITE else agent2
        move = None
        start_turn_time = time.perf_counter()
        try:
            # Give Minimax slightly more time/depth budget for 9MM if feasible
            if isinstance(current_agent, Minimax):
                move = current_agent.get_best_move(state, max_time_seconds=2.0)  # Increased time budget
            # elif isinstance(current_agent, MonteCarloTreeSearch): # Add back if needed
            #      move = current_agent.get_best_move(state, max_time_seconds=1.0) # Example time for MCTS
            else:  # DQN agents
                move = current_agent.get_best_move(state)
        except Exception as e:
            print(f"ERR in {type(current_agent).__name__}.get_best_move: {e}")
            lm_left = board.get_legal_moves(state)
            if not lm_left:
                break
            else:
                move = random.choice(lm_left)
        # turn_time = time.perf_counter() - start_turn_time
        # print(f" Turn {game_moves+1} {state.current_player.name} ({type(current_agent).__name__}) took {turn_time:.3f}s")
        if move is None: break
        state = board.make_move(state, move);
        game_moves += 1
    winner = board.get_winner(state)
    if game_moves >= max_moves and winner == Player.NONE:
        # print(f"Game ended by MAX_MOVES ({max_moves}) limit. Draw.")
        return Player.NONE
    return winner


# --- Main Tournament Logic (run_tournament_9mm - same structure as 6MM version) ---
def run_tournament_9mm(agents_to_test: List[str], num_games: int):
    board = BOARD_CLASS();
    results = {name: {"wins": 0, "losses": 0, "draws": 0, "played_white": 0, "played_black": 0} for name in
               agents_to_test}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"tournament_results_{board.board_size}mm_{timestamp}.txt"
    with open(results_filename, "w") as f_out:
        log = lambda msg: (f_out.write(msg + "\n"), print(msg))
        log(f"Tournament 9MM: {timestamp}\nBoard: {BOARD_CLASS.__name__}, Games/matchup: {num_games}\nAgents: {', '.join(agents_to_test)}\n")
        for i in range(len(agents_to_test)):
            for j in range(i, len(agents_to_test)):
                a1_name, a2_name = agents_to_test[i], agents_to_test[j]
                if a1_name == a2_name and len(agents_to_test) > 1: continue
                log(f"--- Matchup: {a1_name} vs {a2_name} ---")
                stats = {"w1_white": 0, "w2_white": 0, "draws": 0}  # Simplified tracking for wins as white

                # Agent1 White
                n_half1 = num_games // 2
                if n_half1 > 0:
                    log(f"  {n_half1} games: {a1_name}(W) vs {a2_name}(B)...")
                    ag1 = get_agent_instance(a1_name, board, Player.WHITE);
                    ag2 = get_agent_instance(a2_name, board, Player.BLACK)
                    if not ag1 or not ag2: log("    Skip: agent init error"); continue
                    for k in range(n_half1):
                        winner = play_game(board, ag1, ag2)
                        if winner == Player.WHITE:
                            results[a1_name]["wins"] += 1; results[a2_name]["losses"] += 1; stats["w1_white"] += 1
                        elif winner == Player.BLACK:
                            results[a2_name]["wins"] += 1; results[a1_name]["losses"] += 1
                        else:
                            results[a1_name]["draws"] += 1; results[a2_name]["draws"] += 1; stats["draws"] += 1
                        results[a1_name]["played_white"] += 1;
                        results[a2_name]["played_black"] += 1
                        if (k + 1) % (max(1, n_half1 // 2)) == 0: print(
                            f"    Game {k + 1}/{n_half1} done.")  # Log more often
                # Agent2 White
                n_half2 = num_games - n_half1
                if a1_name != a2_name and n_half2 > 0:
                    log(f"  {n_half2} games: {a2_name}(W) vs {a1_name}(B)...")
                    ag1_sw = get_agent_instance(a1_name, board, Player.BLACK);
                    ag2_sw = get_agent_instance(a2_name, board, Player.WHITE)
                    if not ag1_sw or not ag2_sw: log("    Skip swapped: agent init error"); continue
                    for k in range(n_half2):
                        winner = play_game(board, ag2_sw, ag1_sw)
                        if winner == Player.WHITE:
                            results[a2_name]["wins"] += 1; results[a1_name]["losses"] += 1; stats["w2_white"] += 1
                        elif winner == Player.BLACK:
                            results[a1_name]["wins"] += 1; results[a2_name]["losses"] += 1
                        else:
                            results[a1_name]["draws"] += 1; results[a2_name]["draws"] += 1; stats["draws"] += 1
                        results[a2_name]["played_white"] += 1;
                        results[a1_name]["played_black"] += 1
                        if (k + 1) % (max(1, n_half2 // 2)) == 0: print(f"    Game {k + 1}/{n_half2} done.")
                log(f"  Matchup Totals ({a1_name} vs {a2_name}): {a1_name} wins (as W): {stats['w1_white']}, {a2_name} wins (as W): {stats['w2_white']}, Draws: {stats['draws']}\n")
        log("\n--- Overall Tournament Results (9MM) ---")
        for name, stat_vals in results.items():
            tot = stat_vals["wins"] + stat_vals["losses"] + stat_vals["draws"]
            wr = (stat_vals["wins"] / tot * 100) if tot > 0 else 0
            log(f"Agent: {name}\n  W: {stat_vals['wins']}, L: {stat_vals['losses']}, D: {stat_vals['draws']}\n  WR: {wr:.2f}%\n  Played W: {stat_vals['played_white']}, B: {stat_vals['played_black']}\n--------------------")
        log(f"Results saved to {results_filename}")


if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR, exist_ok=True)
    agents_for_9mm_tournament = [
        "Random", "Minimax-L1", "Minimax-L2",  # Higher levels likely too slow
        "RLDQNAgent", "MDQNAgent"
    ]
    # agents_for_9mm_tournament = ["Random", "Minimax-L1"] # Very quick test
    run_tournament_9mm(agents_for_9mm_tournament, NUM_GAMES_PER_MATCHUP)

# --- END OF FILE run_tournament_9mm.py ---