import random
import sys
import os
import pygame
import time # Import time for potential delays

from game.GamePhase import GamePhase
# Adjust imports based on how you run the project.
# If running UI/PyGame.py directly, ensure parent dirs are in PYTHONPATH
# Or use relative imports if running as part of a package.
# Assuming running from root or appropriate PYTHONPATH setup:
from game.util.MoveType import MoveType
from game.util.Player import Player
from game.board.Board import Board # Import base class
from game.board.NineMensMorrisBoard import NineMensMorrisBoard
from game.board.ThreeMensMorrisBoard import ThreeMensMorrisBoard
from game.board.SixMensMorrisBoard import SixMensMorrisBoard
from game.board.TwelveMensMorrisBoard import TwelveMensMorrisBoard
from engines.Minimax import Minimax
from engines.MonteCarlo import MonteCarloTreeSearch
from engines.Unbitable import GraphAgent # For Graph-based AI
from engines.DQN_Agent import DQNAgent # For DQN Agent
from game.board.Maps import NINE_MEN_MILLS, SIX_MEN_MILLS, THREE_MEN_MILLS, TWELVE_MEN_MILLS

# --- Constants ---
SCREEN_WIDTH = 1536
SCREEN_HEIGHT = 864
FPS = 30

# --- Colors ---
WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)
# Theme Colors (will be set in main)
BgColor = None
TextColor = None
HighlightColor = None
# Color Themes
WHITE_THEME = [(210, 180, 140), (18, 159, 184), (173, 88, 2)] # BG, Highlight, Text
BLACK_THEME = [(30, 30, 30), (0, 200, 0), (200, 200, 200)] # BG, Highlight, Text
RETRO_THEME = [(0, 0, 255), (255, 0, 0), (0, 255, 255)] # BG, Highlight, Text

# --- Font ---
# Robust path handling for the font file
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    FONT_PATH = os.path.join(current_dir, "font.ttf")
    if not os.path.exists(FONT_PATH):
        # Fallback if font isn't in UI directory (maybe it's in root?)
        FONT_PATH = os.path.join(os.path.dirname(current_dir), "font.ttf")
    if not os.path.exists(FONT_PATH):
         raise FileNotFoundError("Font file not found")
except Exception as e:
    print(f"Error finding font: {e}. Using default pygame font.")
    FONT_PATH = None # Use pygame default font

# --- Position Coordinates (Centered) ---
# (Coordinates definitions remain the same as provided before)
nine_mens_coords = {
    0:  (50, 50), 1: (300, 50), 2: (550, 50), 3: (100, 100), 4: (300, 100),
    5:  (500, 100), 6: (150, 150), 7: (300, 150), 8: (450, 150), 9: (50, 300),
    10: (100, 300), 11: (150, 300), 12: (450, 300), 13: (500, 300), 14: (550, 300),
    15: (150, 450), 16: (300, 450), 17: (450, 450), 18: (100, 500), 19: (300, 500),
    20: (500, 500), 21: (50, 550), 22: (300, 550), 23: (550, 550)
}
three_mens_coords = {
    0: (150, 150), 1: (300, 150), 2: (450, 150), 3: (150, 300), 4: (300, 300),
    5: (450, 300), 6: (150, 450), 7: (300, 450), 8: (450, 450)
}
six_mens_coords = {
    0:  (50, 50), 1: (300, 50), 2: (550, 50), 3: (150, 150), 4: (300, 150),
    5:  (450, 150), 6: (50, 300), 7: (150, 300), 8: (450, 300), 9: (550, 300),
    10: (150, 450), 11: (300, 450), 12: (450, 450), 13: (50, 550), 14: (300, 550),
    15: (550, 550)
}
twelve_mens_coords = { # Same layout as nine men's for this example
    0:  (50, 50), 1: (300, 50), 2: (550, 50), 3: (100, 100), 4: (300, 100),
    5:  (500, 100), 6: (150, 150), 7: (300, 150), 8: (450, 150), 9: (50, 300),
    10: (100, 300), 11: (150, 300), 12: (450, 300), 13: (500, 300), 14: (550, 300),
    15: (150, 450), 16: (300, 450), 17: (450, 450), 18: (100, 500), 19: (300, 500),
    20: (500, 500), 21: (50, 550), 22: (300, 550), 23: (550, 550)
}

def center_x_positions(coord_dict):
    """Horizontally centers the coordinates dict based on screen width."""
    if not coord_dict: return {}
    xs = [pos[0] for pos in coord_dict.values()]
    min_x, max_x = min(xs), max(xs)
    center_of_coords = (min_x + max_x) // 2
    # Adjust offset calculation for potentially smaller screen widths
    target_center = SCREEN_WIDTH // 2
    offset = target_center - center_of_coords
    new_dict = {}
    for k, (x, y) in coord_dict.items():
        new_dict[k] = (x + offset, y + 50) # Add padding from top
    return new_dict

# Center all coordinate sets
nine_mens_coords = center_x_positions(nine_mens_coords)
three_mens_coords = center_x_positions(three_mens_coords)
six_mens_coords = center_x_positions(six_mens_coords)
twelve_mens_coords = center_x_positions(twelve_mens_coords)

def get_position_coordinates(board_obj):
    """Returns the appropriate coordinate dictionary for the given board instance."""
    if isinstance(board_obj, NineMensMorrisBoard):
        return nine_mens_coords
    elif isinstance(board_obj, ThreeMensMorrisBoard):
        return three_mens_coords
    elif isinstance(board_obj, SixMensMorrisBoard):
        return six_mens_coords
    elif isinstance(board_obj, TwelveMensMorrisBoard):
        return twelve_mens_coords
    print("Warning: Unknown board type for coordinates, defaulting to Nine Men's Morris.")
    return nine_mens_coords # Default fallback

# --- Game Menu Class ---
class GameMenu:
    def __init__(self, screen):
        self.screen = screen
        try:
            self.font = pygame.font.Font(FONT_PATH, 36)
            self.title_font = pygame.font.Font(FONT_PATH, 48)
        except: # Fallback to default font
             self.font = pygame.font.Font(None, 42) # Slightly larger default
             self.title_font = pygame.font.Font(None, 56)
        self.menu_active = True

        self.board_options = [
            "1. Three Men's Morris",    # Index 0 -> Choice 1
            "2. Six Men's Morris",      # Index 1 -> Choice 2
            "3. Nine Men's Morris",     # Index 2 -> Choice 3
            "4. Twelve Men's Morris"    # Index 3 -> Choice 4
        ]
        self.mode_options = [
            "0. Human vs Human",                # Index 0 -> Mode 0
            "1. Human vs AI (You play WHITE)",  # Index 1 -> Mode 1
            "2. AI vs Human (You play BLACK)"   # Index 2 -> Mode 2
        ]
        # Updated difficulty options with clearer names and constraints
        self.difficulty_options = [
            "1. Easy (Minimax 1 ply)",        # Index 0
            "2. Medium (Minimax 2 ply)",      # Index 1
            "3. Hard (Minimax 3 ply)",        # Index 2
            "4. Expert (Minimax 4 ply)",      # Index 3
            "5. Monte Carlo (0.1s)",        # Index 4
            "6. Monte Carlo (1s)",          # Index 5
            "7. Graph AI (3MM Only)",       # Index 6
            "8. DQN Value Agent (3MM Only)" # Index 7
        ]

        # Default selections (e.g., Six Men's Morris, Human vs Human)
        self.selected_board = 1 # Index for Six Men's Morris
        self.selected_mode = 0  # Index for Human vs Human
        self.selected_difficulty = 1 # Default to Medium Minimax if AI mode is chosen

        # Rects for clickable areas
        self.board_rects = []
        self.mode_rects = []
        self.diff_rects = []
        self.start_rect = None
        self.theme_rect = None
        self.close_rect = None

    def draw_menu(self):
        """Draws the menu options on the screen in a single column with adjusted spacing."""
        self.screen.fill(BgColor)
        # Clear rect lists for click detection
        self.board_rects = []
        self.mode_rects = []
        self.diff_rects = []

        # Title
        title = self.title_font.render("Morris Game Configuration", True, TextColor)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 45)) # Slightly higher title
        self.screen.blit(title, title_rect)

        y_offset = 90 # Starting Y position for the first section

        # Font for section titles
        section_font = self.font
        # Slightly smaller font for the options within sections
        try:
             option_font = pygame.font.Font(FONT_PATH, 32) # Reduced size for options
             ai_option_font = pygame.font.Font(FONT_PATH, 30) # Even smaller for AI list
        except: # Fallback
             option_font = pygame.font.Font(None, 38)
             ai_option_font = pygame.font.Font(None, 36)

        # --- Board Selection ---
        board_title = section_font.render("Select Board:", True, HighlightColor)
        board_title_rect = board_title.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
        self.screen.blit(board_title, board_title_rect)
        y_offset += 35 # Reduced space after title

        for idx, text in enumerate(self.board_options):
            color = HighlightColor if idx == self.selected_board else TextColor
            option_text = option_font.render(text, True, color)
            rect = option_text.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
            self.screen.blit(option_text, rect)
            self.board_rects.append(rect)
            y_offset += 32 # Reduced spacing between board options

        y_offset += 25 # Reduced space between sections

        # --- Gameplay Mode Selection ---
        mode_title = section_font.render("Select Gameplay Mode:", True, HighlightColor)
        mode_title_rect = mode_title.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
        self.screen.blit(mode_title, mode_title_rect)
        y_offset += 35 # Reduced space after title

        for idx, text in enumerate(self.mode_options):
            color = HighlightColor if idx == self.selected_mode else TextColor
            option_text = option_font.render(text, True, color)
            rect = option_text.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
            self.screen.blit(option_text, rect)
            self.mode_rects.append(rect)
            y_offset += 32 # Reduced spacing between mode options

        # --- AI Opponent Selection (Conditional) ---
        if self.selected_mode != 0: # Only draw if an AI mode is selected
            y_offset += 25 # Reduced space between sections
            diff_title = section_font.render("Select AI Opponent:", True, HighlightColor)
            diff_title_rect = diff_title.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
            self.screen.blit(diff_title, diff_title_rect)
            y_offset += 35 # Reduced space after title

            for idx, text in enumerate(self.difficulty_options):
                color = HighlightColor if idx == self.selected_difficulty else TextColor
                # Use the smaller font for AI options
                option_text = ai_option_font.render(text, True, color)
                rect = option_text.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
                self.screen.blit(option_text, rect)
                self.diff_rects.append(rect)
                y_offset += 30 # Further reduced spacing for AI options
        else:
             # No AI options, y_offset remains at its current value
             pass

        # --- Action Buttons (Positioned at the bottom) ---
        # Place buttons at fixed positions from the bottom, ensuring space above them
        button_base_y = SCREEN_HEIGHT - 40 # Y position for the bottom button (CLOSE)
        button_spacing = 45 # Vertical space between buttons

        close_text = section_font.render("CLOSE", True, TextColor)
        self.close_rect = close_text.get_rect(center=(SCREEN_WIDTH // 2, button_base_y))
        if self.close_rect:
            self.screen.blit(close_text, self.close_rect)

        theme_text = section_font.render("TOGGLE THEME", True, TextColor)
        self.theme_rect = theme_text.get_rect(center=(SCREEN_WIDTH // 2, button_base_y - button_spacing))
        if self.theme_rect:
            self.screen.blit(theme_text, self.theme_rect)

        start_text = section_font.render("START GAME", True, TextColor)
        self.start_rect = start_text.get_rect(center=(SCREEN_WIDTH // 2, button_base_y - 2 * button_spacing))
        if self.start_rect:
            pygame.draw.rect(self.screen, HighlightColor, self.start_rect.inflate(20, 10), 2, border_radius=5) # Border
            self.screen.blit(start_text, self.start_rect)


        pygame.display.flip()

    def run(self):
        """Runs the menu loop, returning the selected options."""
        clock = pygame.time.Clock()
        while self.menu_active:
            self.draw_menu()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    # Check board options
                    for idx, rect in enumerate(self.board_rects):
                        if rect.collidepoint(pos):
                            self.selected_board = idx
                    # Check mode options
                    for idx, rect in enumerate(self.mode_rects):
                        if rect.collidepoint(pos):
                            self.selected_mode = idx
                    # Check difficulty options (if visible)
                    if self.selected_mode != 0:
                        for idx, rect in enumerate(self.diff_rects):
                            if rect.collidepoint(pos):
                                self.selected_difficulty = idx
                    # Check action buttons
                    if self.start_rect and self.start_rect.collidepoint(pos):
                        # Validate choices (e.g., 3MM for specific AIs) before starting
                        if self.selected_difficulty in [6, 7] and self.selected_board != 0: # Index 6=GraphAI, 7=DQN
                            print("Warning: Graph AI and DQN Agent are intended for 3 Men's Morris only.")
                            # Optionally prevent starting or show a warning popup
                            # For now, we allow starting but it might use fallback AI
                        self.menu_active = False
                    if self.theme_rect and self.theme_rect.collidepoint(pos):
                        self.toggle_theme()
                    if self.close_rect and self.close_rect.collidepoint(pos):
                        pygame.quit()
                        sys.exit()

            clock.tick(FPS)
        # Return board choice (1-4), mode choice (0-2), difficulty index (0-7)
        return self.selected_board + 1, self.selected_mode, self.selected_difficulty

    def toggle_theme(self):
        """Cycles through the available color themes."""
        global BgColor, TextColor, HighlightColor
        if BgColor == BLACK_THEME[0]:
            BgColor, HighlightColor, TextColor = WHITE_THEME[0], WHITE_THEME[1], WHITE_THEME[2]
        elif BgColor == WHITE_THEME[0]:
            BgColor, HighlightColor, TextColor = RETRO_THEME[0], RETRO_THEME[1], RETRO_THEME[2]
        else: # Default to BLACK_THEME if current is RETRO or unknown
            BgColor, HighlightColor, TextColor = BLACK_THEME[0], BLACK_THEME[1], BLACK_THEME[2]

# --- Game GUI Class ---
class GameGUI:
    def __init__(self, board_choice, ai_mode, ai_difficulty_index):
        self.board_choice = board_choice
        self.mode_choice = ai_mode
        self.ai_difficulty_index = ai_difficulty_index # Store the index

        # Board setup based on user choice
        standard_boards = {
            1: ThreeMensMorrisBoard,
            2: SixMensMorrisBoard,
            3: NineMensMorrisBoard,
            4: TwelveMensMorrisBoard
        }
        SelectedBoardClass = standard_boards.get(board_choice, NineMensMorrisBoard) # Default to 9MM
        self.board = SelectedBoardClass()
        self.state = self.board.get_initial_board_state()
        print(f"Initialized board: {SelectedBoardClass.__name__}")

        # Player and AI setup
        self.players = [Player.WHITE, Player.BLACK]
        self.ai_player = Player.NONE # Default for Human vs Human
        self.ai = None # AI agent instance
        self.ai_time_limit = 0.0 # Relevant mainly for MCTS

        if ai_mode == 1: # Human (WHITE) vs AI (BLACK)
            self.ai_player = Player.BLACK
            print("Mode: Human (WHITE) vs AI (BLACK)")
        elif ai_mode == 2: # AI (WHITE) vs Human (BLACK)
            self.ai_player = Player.WHITE
            print("Mode: AI (WHITE) vs Human (BLACK)")
        else:
             print("Mode: Human (WHITE) vs Human (BLACK)")

        # Instantiate the correct AI agent based on difficulty index
        if self.ai_player != Player.NONE:
            print(f"AI Difficulty Index selected: {ai_difficulty_index}")
            if ai_difficulty_index in [0, 1, 2, 3]:  # Minimax
                depth = ai_difficulty_index + 1
                self.ai = Minimax(self.board, depth)
                print(f"Initialized Minimax AI with depth {depth}")
            elif ai_difficulty_index == 4:  # Monte Carlo (0.1s)
                self.ai = MonteCarloTreeSearch(self.board)
                self.ai_time_limit = 0.1
                print("Initialized Monte Carlo AI (0.1s)")
            elif ai_difficulty_index == 5:  # Monte Carlo (1s)
                self.ai = MonteCarloTreeSearch(self.board)
                self.ai_time_limit = 1.0
                print("Initialized Monte Carlo AI (1.0s)")
            elif ai_difficulty_index == 6:  # Graph-based AI (Unbitable)
                if isinstance(self.board, ThreeMensMorrisBoard):
                    script_dir = os.path.dirname(__file__)
                    graph_path_rel = os.path.join("..", "states_graph", "Threegame_graph_evaluated.txt")
                    graph_path = os.path.abspath(os.path.join(script_dir, graph_path_rel))
                    if os.path.exists(graph_path):
                         self.ai = GraphAgent(self.board, graph_path)
                         print(f"Initialized Graph-based AI using {graph_path}")
                    else:
                         print(f"ERROR: Evaluated graph file not found at {graph_path}. Using Fallback AI (Minimax Medium).")
                         self.ai = Minimax(self.board, 2) # Fallback
                else:
                    print("Graph AI is only available for Three Men's Morris board. Using Fallback AI (Minimax Medium).")
                    self.ai = Minimax(self.board, 2) # Fallback
            elif ai_difficulty_index == 7: # DQN Value Agent
                 if isinstance(self.board, ThreeMensMorrisBoard):
                     script_dir = os.path.dirname(__file__)
                     model_rel_path = os.path.join("..", "models", "dqn_3mm_value_model.pth")
                     model_path = os.path.abspath(os.path.join(script_dir, model_rel_path))
                     # Ensure the directory exists for the agent to potentially load from
                     os.makedirs(os.path.dirname(model_path), exist_ok=True)
                     self.ai = DQNAgent(self.board, model_path=model_path)
                     print(f"Initialized DQN Agent. Attempting to load model: {model_path}")
                 else:
                     print("DQN Value Agent is only available for Three Men's Morris board. Using Fallback AI (Minimax Medium).")
                     self.ai = Minimax(self.board, 2) # Fallback
            else: # Fallback for unexpected index
                 print(f"Warning: Invalid AI choice index ({ai_difficulty_index}), defaulting to Medium Minimax.")
                 self.ai = Minimax(self.board, 2)

        # Pygame setup
        self.screen = pygame.display.get_surface() # Get screen from main setup
        self.clock = pygame.time.Clock()
        self.coords = get_position_coordinates(self.board)
        self.selected_pos = None # Stores the ID of the piece selected for a MOVE action
        self.running = True # Game loop control
        self.game_over_message = None # Stores winner message

        # Font for messages
        try:
            self.info_font = pygame.font.Font(FONT_PATH, 30)
            self.end_font = pygame.font.Font(FONT_PATH, 48)
        except: # Fallback
             self.info_font = pygame.font.Font(None, 36)
             self.end_font = pygame.font.Font(None, 56)


    def draw_board(self):
        """Draws the current state of the game board."""
        self.screen.fill(BgColor)

        if not self.coords:
            error_text = self.end_font.render("ERROR: Board coordinates not found!", True, (255, 0, 0))
            self.screen.blit(error_text, (50, 50))
            pygame.display.flip()
            return

        line_thickness = 4
        # Map definitions (assuming these are tuples of lines/mills)
        line_map = None
        if isinstance(self.board, NineMensMorrisBoard): line_map = NINE_MEN_MILLS
        elif isinstance(self.board, ThreeMensMorrisBoard): line_map = THREE_MEN_MILLS
        elif isinstance(self.board, SixMensMorrisBoard): line_map = SIX_MEN_MILLS
        elif isinstance(self.board, TwelveMensMorrisBoard): line_map = TWELVE_MEN_MILLS

        # Draw lines connecting board points based on mills
        if line_map:
            drawn_lines = set() # To avoid drawing lines twice for intersecting mills
            for mill in line_map:
                # Mills can be more than 3 points in some complex layouts, draw segments
                for i in range(len(mill) - 1):
                    p1_id, p2_id = mill[i], mill[i+1]
                    # Ensure line is drawn only once using sorted tuple
                    line_key = tuple(sorted((p1_id, p2_id)))
                    if line_key not in drawn_lines and p1_id in self.coords and p2_id in self.coords:
                         # Check connection exists in board graph for robustness
                         if self.board.graph.hasEdge(p1_id, p2_id):
                            pygame.draw.line(self.screen, TextColor, self.coords[p1_id], self.coords[p2_id], line_thickness)
                            drawn_lines.add(line_key)
            # Special case lines for Six Men's Morris not covered by mills
            if isinstance(self.board, SixMensMorrisBoard):
                 extra_lines = [(1,4), (6,7), (8,9), (11,14)]
                 for p1_id, p2_id in extra_lines:
                     line_key = tuple(sorted((p1_id, p2_id)))
                     if line_key not in drawn_lines and p1_id in self.coords and p2_id in self.coords:
                         if self.board.graph.hasEdge(p1_id, p2_id):
                            pygame.draw.line(self.screen, TextColor, self.coords[p1_id], self.coords[p2_id], line_thickness)
                            drawn_lines.add(line_key)

        # Draw board positions and pieces
        piece_radius = 18
        empty_radius = 10
        selection_radius = 25
        for pos_id, pos_coords in self.coords.items():
            player = self.state.get_player_at_position(pos_id)

            # Highlight selected position for MOVE
            if self.selected_pos == pos_id:
                pygame.draw.circle(self.screen, HighlightColor, pos_coords, selection_radius, 3) # Outline

            # Draw piece or empty slot
            if player == Player.WHITE:
                pygame.draw.circle(self.screen, WHITE_COLOR, pos_coords, piece_radius)
                pygame.draw.circle(self.screen, BLACK_COLOR, pos_coords, piece_radius, 1) # Outline
            elif player == Player.BLACK:
                pygame.draw.circle(self.screen, BLACK_COLOR, pos_coords, piece_radius)
            else: # Player.NONE
                pygame.draw.circle(self.screen, TextColor, pos_coords, empty_radius)

        # Draw game status text
        player_name = self.state.current_player.name
        phase_name = self.state.get_current_phase_for_player(self.state.current_player).name
        status_text = f"{player_name}'s Turn ({phase_name})"
        if self.state.need_to_remove_piece:
            status_text += " - REMOVE Opponent's Piece"
        elif self.selected_pos is not None:
             status_text += f" - Move piece from {self.selected_pos}"

        info_surface = self.info_font.render(status_text, True, TextColor)
        info_rect = info_surface.get_rect(bottomleft=(50, SCREEN_HEIGHT - 20))
        self.screen.blit(info_surface, info_rect)

        # Display game over message if applicable
        if self.game_over_message:
            end_surface = self.end_font.render(self.game_over_message, True, HighlightColor)
            end_rect = end_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
            # Add a background for better visibility
            bg_rect = end_rect.inflate(40, 20)
            pygame.draw.rect(self.screen, BgColor, bg_rect, border_radius=10)
            pygame.draw.rect(self.screen, TextColor, bg_rect, 3, border_radius=10) # Border
            self.screen.blit(end_surface, end_rect)

        pygame.display.flip()


    def handle_human_input(self):
        """Processes human player input (mouse clicks). Returns "menu" if pause menu exits, else None."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    action = show_pause_screen(self.screen, self.info_font, self.end_font) # Pass necessary resources
                    if action == "resume":
                        return None # Continue game
                    elif action == "play_again":
                        # Signal to main loop to restart with same settings
                        return "play_again"
                    elif action == "menu":
                        # Signal to main loop to go back to menu
                        return "menu"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1: # Left mouse button
                    mouse_pos = pygame.mouse.get_pos()
                    clicked_id = None
                    # Find which position (if any) was clicked
                    for pos_id, pos_coords in self.coords.items():
                        dx = mouse_pos[0] - pos_coords[0]
                        dy = mouse_pos[1] - pos_coords[1]
                        # Check if click is within ~20 pixels of the center
                        if dx*dx + dy*dy <= 20*20:
                            clicked_id = pos_id
                            break

                    if clicked_id is not None:
                        self.process_click(clicked_id)
        return None # No action needed from main loop

    def process_click(self, clicked_id):
        """Handles the logic for processing a click on a board position."""
        current_player = self.state.current_player
        legal_moves = self.board.get_legal_moves(self.state)

        # 1. Handle REMOVE move
        if self.state.need_to_remove_piece:
            for move in legal_moves:
                if move.move_type == MoveType.REMOVE and move.remove_checker_from_position.id == clicked_id:
                    print(f"Executing REMOVE at {clicked_id}")
                    self.state = self.board.make_move(self.state, move)
                    self.selected_pos = None # Clear selection
                    return # Move completed

        # 2. Handle PLACE move
        #    Corrected line below: removed "game.GamePhase." prefix
        elif self.state.get_current_phase_for_player(current_player) == GamePhase.PLACEMENT:
            for move in legal_moves:
                if move.move_type == MoveType.PLACE and move.to_position.id == clicked_id:
                    print(f"Executing PLACE at {clicked_id}")
                    self.state = self.board.make_move(self.state, move)
                    self.selected_pos = None
                    return

        # 3. Handle MOVE or FLYING move
        #    Corrected line below: removed "game.GamePhase." prefix
        elif self.state.get_current_phase_for_player(current_player) in [GamePhase.MOVEMENT, GamePhase.FLYING]:
            # If no piece is selected yet
            if self.selected_pos is None:
                # Check if the clicked position has the current player's piece
                if self.state.get_player_at_position(clicked_id) == current_player:
                    # Check if this piece *can* be moved (has valid 'from' moves)
                    can_move_from = any(m.from_position.id == clicked_id for m in legal_moves if m.move_type == MoveType.MOVE)
                    if can_move_from:
                         print(f"Selected piece at {clicked_id}")
                         self.selected_pos = clicked_id
                    else:
                         print(f"Piece at {clicked_id} cannot be moved.")
                else:
                    print(f"Clicked on invalid position {clicked_id} to start move.")
            # If a piece is already selected, try to move it to the clicked position
            else:
                # Check if the clicked position is a valid destination for the selected piece
                found_move = None
                for move in legal_moves:
                    # Check for valid MOVE action matching selection and click
                    if (move.move_type == MoveType.MOVE and
                        move.from_position is not None and # Ensure from_position exists
                        move.from_position.id == self.selected_pos and
                        move.to_position is not None and # Ensure to_position exists
                        move.to_position.id == clicked_id):
                        found_move = move
                        break # Found the move

                if found_move:
                    print(f"Executing MOVE from {self.selected_pos} to {clicked_id}")
                    self.state = self.board.make_move(self.state, found_move)
                    self.selected_pos = None # Clear selection after move
                else:
                    # Invalid destination or clicked on the same piece again - deselect
                    # Also deselect if clicking an opponent's piece or empty space that isn't a valid destination
                    print(f"Invalid destination {clicked_id} or deselected piece at {self.selected_pos}.")
                    self.selected_pos = None

        else:
            # Should not happen in normal gameplay
             print("Clicked during unexpected game phase or state.")
             self.selected_pos = None


    def play_ai_turn(self):
        """Handles the AI's turn."""
        if not self.ai: return # Should not happen if ai_player is set

        # Display "AI is thinking..."
        font = self.info_font
        thinking_text = font.render("AI is thinking...", True, HighlightColor)
        text_rect = thinking_text.get_rect(center=(SCREEN_WIDTH // 2, 30)) # Position at top-center
        # Draw board state before AI moves
        self.draw_board()
        # Draw thinking text with background
        pygame.draw.rect(self.screen, BgColor, text_rect.inflate(10, 5))
        self.screen.blit(thinking_text, text_rect)
        pygame.display.flip()

        ai_move = None
        start_time = time.perf_counter() # More precise timing

        try:
            # Get move from the selected AI
            if isinstance(self.ai, MonteCarloTreeSearch):
                ai_move = self.ai.get_best_move(self.state, self.ai_time_limit)
            elif isinstance(self.ai, Minimax):
                 # Pass state and time limit (Minimax implementation decides how to use it)
                 ai_move = self.ai.get_best_move(self.state, self.ai_time_limit)
            elif isinstance(self.ai, (DQNAgent, GraphAgent)): # Group agents not needing time limit
                ai_move = self.ai.get_best_move(self.state)
            else:
                print(f"Warning: Unknown AI type: {type(self.ai)}. Cannot get move.")

        except Exception as e:
             print(f"Error during AI move calculation: {e}")
             # Fallback: try a random move if possible
             legal_moves = self.board.get_legal_moves(self.state)
             if legal_moves:
                 ai_move = random.choice(legal_moves)
                 print("AI Error: Falling back to random move.")
             else:
                 self.running = False # No moves possible

        end_time = time.perf_counter()
        time_taken = end_time - start_time
        print(f"AI ({type(self.ai).__name__}) took {time_taken:.4f} seconds.")

        # Optional wait if move was too fast for UI feedback
        if time_taken < 0.2:
             pygame.time.wait(int(200 - time_taken * 1000))

        if ai_move is None:
            # This might happen if the game ends exactly on AI's turn start
            # Or if AI truly has no moves (should be caught by game over check)
            print("AI could not find a valid move. Checking game status.")
            if not self.board.check_if_game_is_over(self.state):
                 print("Warning: AI returned None but game is not over!")
                 # Potentially force end or handle error state
                 self.game_over_message = "AI Error - Cannot Move"
            self.running = False # Assume game ends if AI can't move
            return

        # Execute the AI's move
        self.state = self.board.make_move(self.state, ai_move)
        self.selected_pos = None # Reset player selection after AI move


    def run(self):
        """Main game loop."""
        while self.running:
            # Check for game over condition *before* the turn starts
            if self.board.check_if_game_is_over(self.state):
                winner = self.board.get_winner(self.state)
                if winner == Player.NONE:
                    self.game_over_message = "Game Over: It's a Draw!"
                else:
                    self.game_over_message = f"Game Over: {winner.name} Wins!"
                print(self.game_over_message)
                self.running = False # Stop game loop, proceed to end screen handling
                break # Exit loop immediately

            action = None
            if self.state.current_player == self.ai_player:
                self.play_ai_turn()
            else:
                # Handle human input, check if it returns an action string
                action = self.handle_human_input()

            # Draw the updated board state
            self.draw_board()

            # Handle signals from handle_human_input (pause menu actions)
            if action == "menu":
                return "menu" # Signal main to return to menu
            elif action == "play_again":
                return "play_again" # Signal main to restart game

            self.clock.tick(FPS)

        # --- Game Over Sequence ---
        # Keep displaying the final board with the game over message
        self.draw_board() # Ensure final state is drawn with message
        pygame.time.wait(1000) # Pause for 1 second to show the message

        # Show end screen with options
        action = show_end_screen(self.screen, self.info_font, self.end_font, self.game_over_message)

        # Return action selected from the end screen
        return action # Will be "play_again", "menu", or None (if closed)


def show_pause_screen(screen, font, title_font):
    """
    Displays the pause screen and handles user interaction.
    Returns "resume", "play_again", or "menu".
    Needs screen and fonts passed to it.
    """
    paused = True
    while paused:
        screen.fill(BgColor) # Use global BgColor

        pause_text = title_font.render("Paused", True, HighlightColor)
        screen.blit(pause_text, pause_text.get_rect(center=(SCREEN_WIDTH//2, 250)))

        resume_surface = font.render("RESUME (ESC)", True, TextColor)
        resume_rect = resume_surface.get_rect(center=(SCREEN_WIDTH//2, 350))
        pygame.draw.rect(screen, HighlightColor, resume_rect.inflate(20, 10), 2, border_radius=5)
        screen.blit(resume_surface, resume_rect)

        playagain_surface = font.render("PLAY AGAIN", True, TextColor)
        playagain_rect = playagain_surface.get_rect(center=(SCREEN_WIDTH//2, 420))
        screen.blit(playagain_surface, playagain_rect)

        exit_surface = font.render("EXIT TO MENU", True, TextColor)
        exit_rect = exit_surface.get_rect(center=(SCREEN_WIDTH//2, 490))
        screen.blit(exit_surface, exit_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "resume" # Resume game
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if resume_rect.collidepoint(event.pos):
                        return "resume"
                    elif playagain_rect.collidepoint(event.pos):
                        return "play_again"
                    elif exit_rect.collidepoint(event.pos):
                        return "menu"
        pygame.time.Clock().tick(15) # Lower FPS for pause menu


def show_end_screen(screen, font, title_font, message):
    """
    Displays the end game screen with the result and options.
    Returns "play_again", "menu", or None (if window closed).
    Needs screen, fonts, and message passed to it.
    """
    running = True
    while running:
        screen.fill(BgColor)

        # Display the game over message passed to the function
        win_surface = title_font.render(message, True, HighlightColor)
        screen.blit(win_surface, win_surface.get_rect(center=(SCREEN_WIDTH//2, 250)))

        playagain_surface = font.render("PLAY AGAIN", True, TextColor)
        playagain_rect = playagain_surface.get_rect(center=(SCREEN_WIDTH//2, 350))
        pygame.draw.rect(screen, HighlightColor, playagain_rect.inflate(20, 10), 2, border_radius=5)
        screen.blit(playagain_surface, playagain_rect)

        exit_surface = font.render("EXIT TO MENU", True, TextColor)
        exit_rect = exit_surface.get_rect(center=(SCREEN_WIDTH//2, 420))
        screen.blit(exit_surface, exit_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                return None # Signal exit
            elif event.type == pygame.MOUSEBUTTONDOWN:
                 if event.button == 1:
                    if playagain_rect.collidepoint(event.pos):
                        return "play_again"
                    elif exit_rect.collidepoint(event.pos):
                        return "menu"
        pygame.time.Clock().tick(15) # Lower FPS for end screen

# --- Main Function ---
def main():
    global BgColor, TextColor, HighlightColor
    # Default theme
    BgColor, HighlightColor, TextColor = BLACK_THEME[0], BLACK_THEME[1], BLACK_THEME[2]

    # Initialize Pygame
    pygame.init()
    # Set screen size (use constants)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Morris Game")

    while True: # Main application loop (allows returning to menu)
        menu = GameMenu(screen)
        # Get selections from the menu
        board_choice, mode_choice, ai_difficulty_index = menu.run()

        # Create and run the game instance
        game_gui = GameGUI(board_choice, mode_choice, ai_difficulty_index)
        result = game_gui.run() # Run the game loop

        if result == "menu":
            continue # Go back to the start of the main loop to show the menu again
        elif result == "play_again":
             # Re-run the game with the same settings (looping in main already handles this)
             # We just need to let the loop continue to create a new GameGUI instance
             print("Restarting game with same settings...")
             continue
        else:
            # If result is None (closed window during end screen) or unexpected, exit
            break

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    # Ensure the UI directory is in the path if running directly,
    # or manage imports appropriately if run from root.
    # Example: Add parent directory to path if needed
    # script_dir = os.path.dirname(__file__)
    # parent_dir = os.path.dirname(script_dir)
    # if parent_dir not in sys.path:
    #    sys.path.append(parent_dir)

    main()