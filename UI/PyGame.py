# --- START OF FILE UI/PyGame.py ---

import sys
import os
from random import random

import pygame
import time  # Import time for potential delays

# Adjust imports based on how you run the project.
from game.util.MoveType import MoveType
from game.util.Player import Player
from game.board.Board import Board  # Import base class
from game.board.NineMensMorrisBoard import NineMensMorrisBoard
from game.board.ThreeMensMorrisBoard import ThreeMensMorrisBoard
from game.board.SixMensMorrisBoard import SixMensMorrisBoard
from game.board.TwelveMensMorrisBoard import TwelveMensMorrisBoard
from game.GamePhase import GamePhase  # Import GamePhase directly
from engines.Minimax import Minimax
from engines.MonteCarlo import MonteCarloTreeSearch
from engines.Unbitable import GraphAgent
from engines.StateValueAgent import StateValueAgent  # Renamed for clarity
from engines.RL_DQN_Agent import RLDQNAgent  # Agent for Reinforcement Learning
from engines.mDQN_Agent import MDQNAgent  # For Modular DQN
from game.board.Maps import NINE_MEN_MILLS, SIX_MEN_MILLS, THREE_MEN_MILLS, TWELVE_MEN_MILLS

# --- Constants ---
SCREEN_WIDTH = 1536
SCREEN_HEIGHT = 864
FPS = 30

# --- Colors ---
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
BgColor = None
TextColor = None
HighlightColor = None
WHITE_THEME = [(210, 180, 140), (18, 159, 184), (173, 88, 2)]
BLACK_THEME = [(30, 30, 30), (0, 200, 0), (200, 200, 200)]
RETRO_THEME = [(0, 0, 255), (255, 0, 0), (0, 255, 255)]

# --- Font ---
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    FONT_PATH_UI = os.path.join(current_dir, "font.ttf")  # Path specific to UI dir
    if not os.path.exists(FONT_PATH_UI):
        FONT_PATH_ROOT = os.path.join(os.path.dirname(current_dir), "font.ttf")  # Path if font is in root
        if os.path.exists(FONT_PATH_ROOT):
            FONT_PATH = FONT_PATH_ROOT
        else:
            raise FileNotFoundError("Font file not found in UI or root.")
    else:
        FONT_PATH = FONT_PATH_UI
except Exception as e:
    print(f"Error finding font: {e}. Using default pygame font.")
    FONT_PATH = None

# --- Position Coordinates (Centered) ---
nine_mens_coords = {
    0: (50, 50), 1: (300, 50), 2: (550, 50), 3: (100, 100), 4: (300, 100),
    5: (500, 100), 6: (150, 150), 7: (300, 150), 8: (450, 150), 9: (50, 300),
    10: (100, 300), 11: (150, 300), 12: (450, 300), 13: (500, 300), 14: (550, 300),
    15: (150, 450), 16: (300, 450), 17: (450, 450), 18: (100, 500), 19: (300, 500),
    20: (500, 500), 21: (50, 550), 22: (300, 550), 23: (550, 550)
}
three_mens_coords = {
    0: (150, 150), 1: (300, 150), 2: (450, 150), 3: (150, 300), 4: (300, 300),
    5: (450, 300), 6: (150, 450), 7: (300, 450), 8: (450, 450)
}
six_mens_coords = {
    0: (50, 50), 1: (300, 50), 2: (550, 50), 3: (150, 150), 4: (300, 150),
    5: (450, 150), 6: (50, 300), 7: (150, 300), 8: (450, 300), 9: (550, 300),
    10: (150, 450), 11: (300, 450), 12: (450, 450), 13: (50, 550), 14: (300, 550),
    15: (550, 550)
}
twelve_mens_coords = {
    0: (50, 50), 1: (300, 50), 2: (550, 50), 3: (100, 100), 4: (300, 100),
    5: (500, 100), 6: (150, 150), 7: (300, 150), 8: (450, 150), 9: (50, 300),
    10: (100, 300), 11: (150, 300), 12: (450, 300), 13: (500, 300), 14: (550, 300),
    15: (150, 450), 16: (300, 450), 17: (450, 450), 18: (100, 500), 19: (300, 500),
    20: (500, 500), 21: (50, 550), 22: (300, 550), 23: (550, 550)
}


def center_x_positions(coord_dict):
    if not coord_dict: return {}
    xs = [pos[0] for pos in coord_dict.values()]
    min_x, max_x = min(xs), max(xs)
    center_of_coords = (min_x + max_x) // 2
    target_center = SCREEN_WIDTH // 2
    offset = target_center - center_of_coords
    new_dict = {}
    for k, (x, y) in coord_dict.items():
        new_dict[k] = (x + offset, y + 50)
    return new_dict


nine_mens_coords = center_x_positions(nine_mens_coords)
three_mens_coords = center_x_positions(three_mens_coords)
six_mens_coords = center_x_positions(six_mens_coords)
twelve_mens_coords = center_x_positions(twelve_mens_coords)


def get_position_coordinates(board_obj: Board):
    if isinstance(board_obj, NineMensMorrisBoard):
        return nine_mens_coords
    elif isinstance(board_obj, ThreeMensMorrisBoard):
        return three_mens_coords
    elif isinstance(board_obj, SixMensMorrisBoard):
        return six_mens_coords
    elif isinstance(board_obj, TwelveMensMorrisBoard):
        return twelve_mens_coords
    print("Warning: Unknown board type for coordinates, defaulting to Nine Men's Morris.")
    return nine_mens_coords


class GameMenu:
    def __init__(self, screen):
        self.screen = screen
        try:
            self.font = pygame.font.Font(FONT_PATH, 36)
            self.title_font = pygame.font.Font(FONT_PATH, 48)
        except:
            self.font = pygame.font.Font(None, 42)
            self.title_font = pygame.font.Font(None, 56)
        self.menu_active = True
        self.board_options = [
            "1. Three Men's Morris", "2. Six Men's Morris",
            "3. Nine Men's Morris", "4. Twelve Men's Morris"
        ]
        self.mode_options = [
            "0. Human vs Human", "1. Human vs AI (You play WHITE)",
            "2. AI vs Human (You play BLACK)"
        ]
        self.difficulty_options = [
            "1. Easy (Minimax 1 ply)", "2. Medium (Minimax 2 ply)",
            "3. Hard (Minimax 3 ply)", "4. Expert (Minimax 4 ply)",
            "5. Monte Carlo (0.1s)", "6. Monte Carlo (1s)",
            "7. Graph AI (3MM Only)", "8. DQN Value Agent (3MM Trained)",
            "9. RL DQN Agent (3MM Experimental)",  # Index 8
            "10. mDQN Agent (3MM Experimental)"  # Index 9
        ]
        self.selected_board = 0
        self.selected_mode = 0
        self.selected_difficulty = 0
        self.board_rects, self.mode_rects, self.diff_rects = [], [], []
        self.start_rect, self.theme_rect, self.close_rect = None, None, None

    def draw_menu(self):
        self.screen.fill(BgColor);
        self.board_rects, self.mode_rects, self.diff_rects = [], [], []
        title = self.title_font.render("Morris Game Configuration", True, TextColor)
        self.screen.blit(title, title.get_rect(center=(SCREEN_WIDTH // 2, 45)))
        y_offset = 90;
        section_font = self.font
        try:
            option_font = pygame.font.Font(FONT_PATH, 32); ai_option_font = pygame.font.Font(FONT_PATH,
                                                                                             28)  # Slightly smaller AI font
        except:
            option_font = pygame.font.Font(None, 38); ai_option_font = pygame.font.Font(None, 34)
        board_title = section_font.render("Select Board:", True, HighlightColor)
        self.screen.blit(board_title, board_title.get_rect(center=(SCREEN_WIDTH // 2, y_offset)));
        y_offset += 35
        for idx, text in enumerate(self.board_options):
            color = HighlightColor if idx == self.selected_board else TextColor
            option_text_surf = option_font.render(text, True, color)
            rect = option_text_surf.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
            self.screen.blit(option_text_surf, rect);
            self.board_rects.append(rect);
            y_offset += 32
        y_offset += 20
        mode_title = section_font.render("Select Gameplay Mode:", True, HighlightColor)
        self.screen.blit(mode_title, mode_title.get_rect(center=(SCREEN_WIDTH // 2, y_offset)));
        y_offset += 35
        for idx, text in enumerate(self.mode_options):
            color = HighlightColor if idx == self.selected_mode else TextColor
            option_text_surf = option_font.render(text, True, color)
            rect = option_text_surf.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
            self.screen.blit(option_text_surf, rect);
            self.mode_rects.append(rect);
            y_offset += 32
        if self.selected_mode != 0:
            y_offset += 20
            diff_title = section_font.render("Select AI Opponent:", True, HighlightColor)
            self.screen.blit(diff_title, diff_title.get_rect(center=(SCREEN_WIDTH // 2, y_offset)));
            y_offset += 35
            for idx, text in enumerate(self.difficulty_options):
                color = HighlightColor if idx == self.selected_difficulty else TextColor
                option_text_surf = ai_option_font.render(text, True, color)
                rect = option_text_surf.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
                self.screen.blit(option_text_surf, rect);
                self.diff_rects.append(rect);
                y_offset += 28  # Even tighter spacing for AI
        button_base_y = SCREEN_HEIGHT - 40;
        button_spacing = 45
        self.close_rect = section_font.render("CLOSE", True, TextColor).get_rect(
            center=(SCREEN_WIDTH // 2, button_base_y))
        self.screen.blit(section_font.render("CLOSE", True, TextColor), self.close_rect)
        self.theme_rect = section_font.render("TOGGLE THEME", True, TextColor).get_rect(
            center=(SCREEN_WIDTH // 2, button_base_y - button_spacing))
        self.screen.blit(section_font.render("TOGGLE THEME", True, TextColor), self.theme_rect)
        start_text_surf = section_font.render("START GAME", True, TextColor)
        self.start_rect = start_text_surf.get_rect(center=(SCREEN_WIDTH // 2, button_base_y - 2 * button_spacing))
        pygame.draw.rect(self.screen, HighlightColor, self.start_rect.inflate(20, 10), 2, border_radius=5)
        self.screen.blit(start_text_surf, self.start_rect)
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while self.menu_active:
            self.draw_menu()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    for idx, rect in enumerate(self.board_rects):
                        if rect.collidepoint(pos): self.selected_board = idx
                    for idx, rect in enumerate(self.mode_rects):
                        if rect.collidepoint(pos): self.selected_mode = idx
                    if self.selected_mode != 0:
                        for idx, rect in enumerate(self.diff_rects):
                            if rect.collidepoint(pos): self.selected_difficulty = idx
                    if self.start_rect and self.start_rect.collidepoint(pos):
                        if self.selected_difficulty in [6, 7, 8,
                                                        9] and self.selected_board != 0:  # Indices for GraphAI, StateValue, RL DQN, mDQN
                            print(
                                "Warning: Graph AI, StateValue, RL DQN, and mDQN Agents are primarily for 3 Men's Morris.")
                        self.menu_active = False
                    if self.theme_rect and self.theme_rect.collidepoint(pos): self.toggle_theme()
                    if self.close_rect and self.close_rect.collidepoint(pos): pygame.quit(); sys.exit()
            clock.tick(FPS)
        return self.selected_board + 1, self.selected_mode, self.selected_difficulty

    def toggle_theme(self):
        global BgColor, TextColor, HighlightColor
        if BgColor == BLACK_THEME[0]:
            BgColor, HighlightColor, TextColor = WHITE_THEME[0], WHITE_THEME[1], WHITE_THEME[2]
        elif BgColor == WHITE_THEME[0]:
            BgColor, HighlightColor, TextColor = RETRO_THEME[0], RETRO_THEME[1], RETRO_THEME[2]
        else:
            BgColor, HighlightColor, TextColor = BLACK_THEME[0], BLACK_THEME[1], BLACK_THEME[2]


class GameGUI:
    def __init__(self, board_choice, ai_mode, ai_difficulty_index):
        self.board_choice = board_choice;
        self.mode_choice = ai_mode;
        self.ai_difficulty_index = ai_difficulty_index
        standard_boards = {1: ThreeMensMorrisBoard, 2: SixMensMorrisBoard, 3: NineMensMorrisBoard,
                           4: TwelveMensMorrisBoard}
        SelectedBoardClass = standard_boards.get(board_choice, NineMensMorrisBoard)
        self.board = SelectedBoardClass();
        self.state = self.board.get_initial_board_state()
        print(f"Initialized board: {SelectedBoardClass.__name__}")
        self.players = [Player.WHITE, Player.BLACK];
        self.ai_player = Player.NONE;
        self.ai = None;
        self.ai_time_limit = 0.0
        if ai_mode == 1:
            self.ai_player = Player.BLACK; print("Mode: Human (WHITE) vs AI (BLACK)")
        elif ai_mode == 2:
            self.ai_player = Player.WHITE; print("Mode: AI (WHITE) vs Human (BLACK)")
        else:
            print("Mode: Human (WHITE) vs Human (BLACK)")

        if self.ai_player != Player.NONE:
            print(f"AI Difficulty Index selected: {ai_difficulty_index}")
            script_dir = os.path.dirname(__file__);
            root_dir = os.path.dirname(script_dir)
            if ai_difficulty_index in [0, 1, 2, 3]:
                depth = ai_difficulty_index + 1; self.ai = Minimax(self.board, depth); print(f"Minimax depth {depth}")
            elif ai_difficulty_index == 4:
                self.ai = MonteCarloTreeSearch(self.board); self.ai_time_limit = 0.1; print("MCTS (0.1s)")
            elif ai_difficulty_index == 5:
                self.ai = MonteCarloTreeSearch(self.board); self.ai_time_limit = 1.0; print("MCTS (1.0s)")
            elif ai_difficulty_index == 6:  # GraphAI
                if isinstance(self.board, ThreeMensMorrisBoard):
                    graph_path = os.path.join(root_dir, "states_graph", "Threegame_graph_evaluated.txt")
                    if os.path.exists(graph_path):
                        self.ai = GraphAgent(self.board, graph_path); print(f"GraphAI: {graph_path}")
                    else:
                        print(f"ERR: Graph file missing: {graph_path}. Fallback."); self.ai = Minimax(self.board, 2)
                else:
                    print("GraphAI for 3MM only. Fallback."); self.ai = Minimax(self.board, 2)
            elif ai_difficulty_index == 7:  # StateValueAgent
                if isinstance(self.board, ThreeMensMorrisBoard):
                    model_path = os.path.join(root_dir, "models", "3mm_state_value_model.pth")
                    self.ai = StateValueAgent(self.board, model_path=model_path);
                    print(f"StateValueAgent: {model_path}")
                else:
                    print("StateValueAgent for 3MM only. Fallback."); self.ai = Minimax(self.board, 2)
            elif ai_difficulty_index == 8:  # RLDQNAgent
                if isinstance(self.board, ThreeMensMorrisBoard):
                    model_path = os.path.join(root_dir, "models", f"{self.board.board_size}mm_rl_dqn_model.pth")
                    self.ai = RLDQNAgent(self.board, model_path=model_path);
                    print(f"RLDQNAgent: {model_path}")
                else:
                    print("RLDQNAgent for 3MM only. Fallback."); self.ai = Minimax(self.board, 2)
            elif ai_difficulty_index == 9:  # MDQNAgent
                if isinstance(self.board, ThreeMensMorrisBoard):
                    self.ai = MDQNAgent(self.board);
                    print(f"MDQNAgent for 3MM initialized.")  # Models loaded internally
                else:
                    print("MDQNAgent for 3MM only. Fallback."); self.ai = Minimax(self.board, 2)
            else:
                print(f"Invalid AI choice ({ai_difficulty_index}). Fallback."); self.ai = Minimax(self.board, 2)
        self.screen = pygame.display.get_surface();
        self.clock = pygame.time.Clock()
        self.coords = get_position_coordinates(self.board);
        self.selected_pos = None
        self.running = True;
        self.game_over_message = None
        try:
            self.info_font = pygame.font.Font(FONT_PATH, 30); self.end_font = pygame.font.Font(FONT_PATH, 48)
        except:
            self.info_font = pygame.font.Font(None, 36); self.end_font = pygame.font.Font(None, 56)

    def draw_board(self):  # (Same as previous correct version)
        self.screen.fill(BgColor)
        if not self.coords:
            error_text = self.end_font.render("ERROR: Board coordinates not found!", True, (255, 0, 0))
            self.screen.blit(error_text, (50, 50));
            pygame.display.flip();
            return
        line_thickness = 4;
        line_map = None
        if isinstance(self.board, NineMensMorrisBoard):
            line_map = NINE_MEN_MILLS
        elif isinstance(self.board, ThreeMensMorrisBoard):
            line_map = THREE_MEN_MILLS
        elif isinstance(self.board, SixMensMorrisBoard):
            line_map = SIX_MEN_MILLS
        elif isinstance(self.board, TwelveMensMorrisBoard):
            line_map = TWELVE_MEN_MILLS
        if line_map:
            drawn_lines = set()
            for mill_line in line_map:
                for i in range(len(mill_line) - 1):
                    p1_id, p2_id = mill_line[i], mill_line[i + 1]
                    line_key = tuple(sorted((p1_id, p2_id)))
                    if line_key not in drawn_lines and p1_id in self.coords and p2_id in self.coords:
                        if self.board.graph.hasEdge(p1_id, p2_id):
                            pygame.draw.line(self.screen, TextColor, self.coords[p1_id], self.coords[p2_id],
                                             line_thickness)
                            drawn_lines.add(line_key)
                # Check for cycle in mill definition (e.g., first and last point)
                if len(mill_line) > 2 and self.board.graph.hasEdge(mill_line[-1], mill_line[0]):
                    line_key_cycle = tuple(sorted((mill_line[-1], mill_line[0])))
                    if line_key_cycle not in drawn_lines and mill_line[-1] in self.coords and mill_line[
                        0] in self.coords:
                        pygame.draw.line(self.screen, TextColor, self.coords[mill_line[-1]], self.coords[mill_line[0]],
                                         line_thickness)
                        drawn_lines.add(line_key_cycle)
            if isinstance(self.board, SixMensMorrisBoard):
                extra_lines = [(1, 4), (6, 7), (8, 9), (11, 14)]  # Specific to Six Men's Morris visual
                for p1_id, p2_id in extra_lines:
                    line_key = tuple(sorted((p1_id, p2_id)))
                    if line_key not in drawn_lines and p1_id in self.coords and p2_id in self.coords:
                        if self.board.graph.hasEdge(p1_id, p2_id):  # Check actual connection
                            pygame.draw.line(self.screen, TextColor, self.coords[p1_id], self.coords[p2_id],
                                             line_thickness)
                            drawn_lines.add(line_key)
        piece_radius = 18;
        empty_radius = 10;
        selection_radius = 25
        for pos_id, pos_coords in self.coords.items():
            player = self.state.get_player_at_position(pos_id)
            if self.selected_pos == pos_id: pygame.draw.circle(self.screen, HighlightColor, pos_coords,
                                                               selection_radius, 3)
            if player == Player.WHITE:
                pygame.draw.circle(self.screen, WHITE_COLOR, pos_coords, piece_radius); pygame.draw.circle(self.screen,
                                                                                                           BLACK_COLOR,
                                                                                                           pos_coords,
                                                                                                           piece_radius,
                                                                                                           1)
            elif player == Player.BLACK:
                pygame.draw.circle(self.screen, BLACK_COLOR, pos_coords, piece_radius)
            else:
                pygame.draw.circle(self.screen, TextColor, pos_coords, empty_radius)
        player_name = self.state.current_player.name
        phase_name = self.state.get_current_phase_for_player(self.state.current_player).name
        status_text = f"{player_name}'s Turn ({phase_name})"
        if self.state.need_to_remove_piece:
            status_text += " - REMOVE Opponent's Piece"
        elif self.selected_pos is not None:
            status_text += f" - Move piece from {self.selected_pos}"
        info_surface = self.info_font.render(status_text, True, TextColor)
        self.screen.blit(info_surface, info_surface.get_rect(bottomleft=(50, SCREEN_HEIGHT - 20)))
        if self.game_over_message:
            end_surface = self.end_font.render(self.game_over_message, True, HighlightColor)
            end_rect = end_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            bg_rect = end_rect.inflate(40, 20);
            pygame.draw.rect(self.screen, BgColor, bg_rect, border_radius=10)
            pygame.draw.rect(self.screen, TextColor, bg_rect, 3, border_radius=10)
            self.screen.blit(end_surface, end_rect)
        pygame.display.flip()

    def handle_human_input(self):  # (Same as previous correct version)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False; pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    action = show_pause_screen(self.screen, self.info_font, self.end_font)
                    if action == "resume":
                        return None
                    elif action == "play_again":
                        return "play_again"
                    elif action == "menu":
                        return "menu"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = pygame.mouse.get_pos();
                    clicked_id = None
                    for pos_id, pos_coords in self.coords.items():
                        dx, dy = mouse_pos[0] - pos_coords[0], mouse_pos[1] - pos_coords[1]
                        if dx * dx + dy * dy <= 20 * 20: clicked_id = pos_id; break
                    if clicked_id is not None: self.process_click(clicked_id)
        return None

    def process_click(self, clicked_id):  # (Same as previous correct version)
        current_player = self.state.current_player;
        legal_moves = self.board.get_legal_moves(self.state)
        if self.state.need_to_remove_piece:
            for move in legal_moves:
                if move.move_type == MoveType.REMOVE and move.remove_checker_from_position.id == clicked_id:
                    self.state = self.board.make_move(self.state, move);
                    self.selected_pos = None;
                    return
        elif self.state.get_current_phase_for_player(current_player) == GamePhase.PLACEMENT:
            for move in legal_moves:
                if move.move_type == MoveType.PLACE and move.to_position.id == clicked_id:
                    self.state = self.board.make_move(self.state, move);
                    self.selected_pos = None;
                    return
        elif self.state.get_current_phase_for_player(current_player) in [GamePhase.MOVEMENT, GamePhase.FLYING]:
            if self.selected_pos is None:
                if self.state.get_player_at_position(clicked_id) == current_player:
                    if any(m.from_position and m.from_position.id == clicked_id for m in legal_moves if
                           m.move_type == MoveType.MOVE):
                        self.selected_pos = clicked_id
            else:
                found_move = None
                for move in legal_moves:
                    if (
                            move.move_type == MoveType.MOVE and move.from_position and move.from_position.id == self.selected_pos and
                            move.to_position and move.to_position.id == clicked_id):
                        found_move = move;
                        break
                if found_move:
                    self.state = self.board.make_move(self.state, found_move); self.selected_pos = None
                else:
                    self.selected_pos = None

    def play_ai_turn(self):  # (Same as previous correct version)
        if not self.ai: return
        font = self.info_font;
        thinking_text = font.render("AI is thinking...", True, HighlightColor)
        text_rect = thinking_text.get_rect(center=(SCREEN_WIDTH // 2, 30))
        self.draw_board();
        pygame.draw.rect(self.screen, BgColor, text_rect.inflate(10, 5))
        self.screen.blit(thinking_text, text_rect);
        pygame.display.flip()
        ai_move = None;
        start_time = time.perf_counter()
        try:
            if isinstance(self.ai, MonteCarloTreeSearch):
                ai_move = self.ai.get_best_move(self.state, self.ai_time_limit)
            elif isinstance(self.ai, Minimax):
                ai_move = self.ai.get_best_move(self.state, self.ai_time_limit)
            elif isinstance(self.ai, (StateValueAgent, RLDQNAgent, MDQNAgent, GraphAgent)):
                ai_move = self.ai.get_best_move(self.state)
            else:
                print(f"Warning: Unknown AI type: {type(self.ai)}.")
        except Exception as e:
            print(f"Error during AI move: {e}")
            legal_moves = self.board.get_legal_moves(self.state)
            if legal_moves:
                ai_move = random.choice(legal_moves); print("AI Error: Fallback to random.")
            else:
                self.running = False
        end_time = time.perf_counter();
        time_taken = end_time - start_time
        # print(f"AI ({type(self.ai).__name__}) took {time_taken:.4f}s.")
        if time_taken < 0.1: pygame.time.wait(int(100 - time_taken * 1000))
        if ai_move is None:
            if not self.board.check_if_game_is_over(self.state): print("Warning: AI None but game not over!")
            self.running = False;
            return
        self.state = self.board.make_move(self.state, ai_move);
        self.selected_pos = None

    def run(self):  # (Same as previous correct version)
        while self.running:
            if self.board.check_if_game_is_over(self.state):
                winner = self.board.get_winner(self.state)
                self.game_over_message = "Game Over: It's a Draw!" if winner == Player.NONE else f"Game Over: {winner.name} Wins!"
                print(self.game_over_message);
                self.running = False;
                break
            action = None
            if self.state.current_player == self.ai_player and self.ai is not None:
                self.play_ai_turn()
            else:
                action = self.handle_human_input()
            self.draw_board()
            if action == "menu":
                return "menu"
            elif action == "play_again":
                return "play_again"
            self.clock.tick(FPS)
        self.draw_board();
        pygame.time.wait(1000)
        action = show_end_screen(self.screen, self.info_font, self.end_font, self.game_over_message)
        return action


def show_pause_screen(screen, font, title_font):  # (Same as previous correct version)
    paused = True
    while paused:
        screen.fill(BgColor)
        pause_text_surf = title_font.render("Paused", True, HighlightColor)
        screen.blit(pause_text_surf, pause_text_surf.get_rect(center=(SCREEN_WIDTH // 2, 250)))
        resume_surf = font.render("RESUME (ESC)", True, TextColor)
        resume_rect = resume_surf.get_rect(center=(SCREEN_WIDTH // 2, 350))
        pygame.draw.rect(screen, HighlightColor, resume_rect.inflate(20, 10), 2, border_radius=5)
        screen.blit(resume_surf, resume_rect)
        playagain_surf = font.render("PLAY AGAIN", True, TextColor)
        playagain_rect = playagain_surf.get_rect(center=(SCREEN_WIDTH // 2, 420))
        screen.blit(playagain_surf, playagain_rect)
        exit_surf = font.render("EXIT TO MENU", True, TextColor)
        exit_rect = exit_surf.get_rect(center=(SCREEN_WIDTH // 2, 490))
        screen.blit(exit_surf, exit_rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return "resume"
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if resume_rect.collidepoint(event.pos): return "resume"
                if playagain_rect.collidepoint(event.pos): return "play_again"
                if exit_rect.collidepoint(event.pos): return "menu"
        pygame.time.Clock().tick(15)


def show_end_screen(screen, font, title_font, message):  # (Same as previous correct version)
    running = True
    while running:
        screen.fill(BgColor)
        if message is None: message = "Game Ended"
        win_surf = title_font.render(message, True, HighlightColor)
        screen.blit(win_surf, win_surf.get_rect(center=(SCREEN_WIDTH // 2, 250)))
        playagain_surf = font.render("PLAY AGAIN", True, TextColor)
        playagain_rect = playagain_surf.get_rect(center=(SCREEN_WIDTH // 2, 350))
        pygame.draw.rect(screen, HighlightColor, playagain_rect.inflate(20, 10), 2, border_radius=5)
        screen.blit(playagain_surf, playagain_rect)
        exit_surf = font.render("EXIT TO MENU", True, TextColor)
        exit_rect = exit_surf.get_rect(center=(SCREEN_WIDTH // 2, 420))
        screen.blit(exit_surf, exit_rect)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit(); return None
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if playagain_rect.collidepoint(event.pos): return "play_again"
                if exit_rect.collidepoint(event.pos): return "menu"
        pygame.time.Clock().tick(15)


def main():
    global BgColor, TextColor, HighlightColor
    BgColor, HighlightColor, TextColor = BLACK_THEME[0], BLACK_THEME[1], BLACK_THEME[2]
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Morris Game")
    while True:
        menu = GameMenu(screen)
        board_choice, mode_choice, ai_difficulty_index = menu.run()
        game_gui = GameGUI(board_choice, mode_choice, ai_difficulty_index)
        result = game_gui.run()
        if result == "menu":
            continue
        elif result == "play_again":
            continue
        else:
            break
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()

# --- END OF FILE UI/PyGame.py ---