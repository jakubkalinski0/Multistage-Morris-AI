import sys
import os
import pygame
from game.util.MoveType import MoveType
from game.util.Player import Player
from game.board.NineMensMorrisBoard import NineMensMorrisBoard
from game.board.ThreeMensMorrisBoard import ThreeMensMorrisBoard
from game.board.SixMensMorrisBoard import SixMensMorrisBoard
from game.board.TwelveMensMorrisBoard import TwelveMensMorrisBoard
from engines.Minimax import Minimax
from game.board.Maps import NINE_MEN_MILLS,SIX_MEN_MILLS,THREE_MEN_MILLS,TWELVE_MEN_MILLS
from engines.MonteCarlo import MonteCarloTreeSearch  # Import klasy Monte Carlo

SCREEN_WIDTH = 1536 
SCREEN_HEIGHT = 864
FPS = 30

WHITE_COLOR = (255, 255, 255)
BLACK_COLOR = (0, 0, 0)

WHITE_TYPE=[(210, 180, 140),(18, 159, 184),(173, 88, 2)]
BLACK_TYPE=[(30, 30, 30),(0, 200, 0),(100, 100, 100)]
RETRO_TYPE=[(0, 0, 255),(255, 0, 0),(0, 255, 255)]

current_dir = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(current_dir, "font.ttf")

# --- MAPOWANIE POZYCJI ---
nine_mens_coords = {
    0:  (50, 50),    # Pozycja 0: lewy górny róg (outer square)
    1:  (300, 50),   # Pozycja 1: górny środek
    2:  (550, 50),   # Pozycja 2: prawy górny róg
    3:  (100, 100),  # Pozycja 3: lewy górny róg middle square
    4:  (300, 100),  # Pozycja 4: środek górnej strony middle square
    5:  (500, 100),  # Pozycja 5: prawy górny róg middle square
    6:  (150, 150),  # Pozycja 6: lewy górny róg inner square
    7:  (300, 150),  # Pozycja 7: górny środek inner square
    8:  (450, 150),  # Pozycja 8: prawy górny róg inner square
    9:  (50, 300),   # Pozycja 9: lewy środek outer square (główna pionowa oś)
    10: (100, 300),  # Pozycja 10: lewy środek middle square
    11: (150, 300),  # Pozycja 11: lewy środek inner square
    12: (450, 300),  # Pozycja 12: prawy środek inner square
    13: (500, 300),  # Pozycja 13: prawy środek middle square
    14: (550, 300),  # Pozycja 14: prawy środek outer square
    15: (150, 450),  # Pozycja 15: dolny środek inner square (lewa strona)
    16: (300, 450),  # Pozycja 16: dolny środek inner square (środek)
    17: (450, 450),  # Pozycja 17: dolny środek inner square (prawa strona)
    18: (100, 500),  # Pozycja 18: lewy dolny róg middle square
    19: (300, 500),  # Pozycja 19: dolny środek middle square
    20: (500, 500),  # Pozycja 20: prawy dolny róg middle square
    21: (50, 550),   # Pozycja 21: lewy dolny róg outer square
    22: (300, 550),  # Pozycja 22: dolny środek outer square
    23: (550, 550)   # Pozycja 23: prawy dolny róg outer square
}


three_mens_coords = {
    0: (150, 150), 1: (300, 150), 2: (450, 150),
    3: (150, 300), 4: (300, 300), 5: (450, 300),
    6: (150, 450), 7: (300, 450), 8: (450, 450)
}

six_mens_coords = {
    0:  (50, 50),    # Pozycja 0 – górny lewy róg
    1:  (300, 50),   # Pozycja 1 – górny środek
    2:  (550, 50),   # Pozycja 2 – górny prawy róg

    3:  (150, 150),  # Pozycja 3 – lewy górny róg górnego mniejszego kwadratu
    4:  (300, 150),  # Pozycja 4 – środkowy
    5:  (450, 150),  # Pozycja 5 – prawy górny róg górnego mniejszego kwadratu

    6:  (50, 300),   # Pozycja 6 – lewy róg zewnętrznego kwadratu
    7:  (150, 300),  # Pozycja 7 – wewnętrzna lewa
    8:  (450, 300),  # Pozycja 8 – wewnętrzna prawa
    9:  (550, 300),  # Pozycja 9 – prawy róg zewnętrznego kwadratu

    10: (150, 450),  # Pozycja 10 – lewy dolny róg środkowego mniejszego kwadratu
    11: (300, 450),  # Pozycja 11 – dolny środek
    12: (450, 450),  # Pozycja 12 – prawy dolny róg środkowego mniejszego kwadratu

    13: (50, 550),   # Pozycja 13 – lewy dolny róg
    14: (300, 550),  # Pozycja 14 – dolny środek
    15: (550, 550)   # Pozycja 15 – prawy dolny róg
}

twelve_mens_coords = {
    0:  (50, 50),    # Pozycja 0: lewy górny róg (outer square)
    1:  (300, 50),   # Pozycja 1: górny środek
    2:  (550, 50),   # Pozycja 2: prawy górny róg
    3:  (100, 100),  # Pozycja 3: lewy górny róg middle square
    4:  (300, 100),  # Pozycja 4: środek górnej strony middle square
    5:  (500, 100),  # Pozycja 5: prawy górny róg middle square
    6:  (150, 150),  # Pozycja 6: lewy górny róg inner square
    7:  (300, 150),  # Pozycja 7: górny środek inner square
    8:  (450, 150),  # Pozycja 8: prawy górny róg inner square
    9:  (50, 300),   # Pozycja 9: lewy środek outer square (główna pionowa oś)
    10: (100, 300),  # Pozycja 10: lewy środek middle square
    11: (150, 300),  # Pozycja 11: lewy środek inner square
    12: (450, 300),  # Pozycja 12: prawy środek inner square
    13: (500, 300),  # Pozycja 13: prawy środek middle square
    14: (550, 300),  # Pozycja 14: prawy środek outer square
    15: (150, 450),  # Pozycja 15: dolny środek inner square (lewa strona)
    16: (300, 450),  # Pozycja 16: dolny środek inner square (środek)
    17: (450, 450),  # Pozycja 17: dolny środek inner square (prawa strona)
    18: (100, 500),  # Pozycja 18: lewy dolny róg middle square
    19: (300, 500),  # Pozycja 19: dolny środek middle square
    20: (500, 500),  # Pozycja 20: prawy dolny róg middle square
    21: (50, 550),   # Pozycja 21: lewy dolny róg outer square
    22: (300, 550),  # Pozycja 22: dolny środek outer square
    23: (550, 550)   # Pozycja 23: prawy dolny róg outer square
}

def center_x_positions(coord_dict):
    xs = [pos[0] for pos in coord_dict.values()]
    min_x, max_x = min(xs), max(xs)
    center_of_coords = (min_x + max_x) // 2
    offset = (SCREEN_WIDTH // 2) - center_of_coords
    new_dict = {}
    for k, (x, y) in coord_dict.items():
        new_dict[k] = (x + offset, y+50)
    return new_dict

# Wycentrowanie wszystkich plansz w poziomie:
nine_mens_coords = center_x_positions(nine_mens_coords)
three_mens_coords = center_x_positions(three_mens_coords)
six_mens_coords = center_x_positions(six_mens_coords)
twelve_mens_coords = center_x_positions(twelve_mens_coords)

def get_position_coordinates(board_obj):
    if isinstance(board_obj, NineMensMorrisBoard):
        return nine_mens_coords
    elif isinstance(board_obj, ThreeMensMorrisBoard):
        return three_mens_coords
    elif isinstance(board_obj, SixMensMorrisBoard):
        return six_mens_coords
    elif isinstance(board_obj, TwelveMensMorrisBoard):
        return twelve_mens_coords
    # Domyślnie zwracamy Nine Men's Morris dla bezpieczeństwa
    return nine_mens_coords

# --- KLASA MENU GRY ---
class GameMenu:
    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.Font(FONT_PATH, 36)
        self.menu_active = True

        self.board_options = [
            "1. Three Men's Morris",
            "2. Six Men's Morris",
            "3. Nine Men's Morris",
            "4. Twelve Men's Morris"
        ]
        self.mode_options = [
            "0. Human vs Human",
            "1. Human vs AI (You play as WHITE)",
            "2. AI vs Human (You play as BLACK)"
        ]
        self.difficulty_options = [
            "1. Easy (1 ply)",
            "2. Medium (2 ply)",
            "3. Hard (3 ply)",
            "4. Expert (4 ply)",
            "5. Monte Carlo (0.1s)",
            "6. Monte Carlo (1s)",
            "7. Graph-based AI"  # Add this new option
        ]

        self.selected_board = 2
        self.selected_mode = 0
        self.selected_difficulty = 2

        # Przechowuje klikowalne recty
        self.board_rects = []
        self.mode_rects = []
        self.diff_rects = []

    def draw_menu(self):
        self.screen.fill(BgColor)
        self.board_rects = []
        self.mode_rects = []
        self.diff_rects = []

        title_font = pygame.font.Font(FONT_PATH, 48)
        title = title_font.render("Morris Game by Student Debil:", True, TextColor)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 40))  # <-- wycentrowanie w poziomie
        self.screen.blit(title, title_rect)

        y = 120
        board_title = self.font.render("Select board:", True, TextColor)
        board_title_rect = board_title.get_rect(center=(SCREEN_WIDTH // 2, y))
        self.screen.blit(board_title, board_title_rect)
        y += 40

        for idx, text in enumerate(self.board_options):
            color = HighlightColor if idx == self.selected_board else TextColor
            option_text = self.font.render(text, True, color)
            rect = option_text.get_rect(center=(SCREEN_WIDTH // 2, y))  # <-- centrowanie
            self.screen.blit(option_text, rect)
            self.board_rects.append(rect)
            y += 32

        y += 40
        mode_title = self.font.render("Select gameplay mode:", True, TextColor)
        mode_title_rect = mode_title.get_rect(center=(SCREEN_WIDTH // 2, y))
        self.screen.blit(mode_title, mode_title_rect)
        y += 40

        for idx, text in enumerate(self.mode_options):
            color = HighlightColor if idx == self.selected_mode else TextColor
            option_text = self.font.render(text, True, color)
            rect = option_text.get_rect(center=(SCREEN_WIDTH // 2, y))  # <-- centrowanie
            self.screen.blit(option_text, rect)
            self.mode_rects.append(rect)
            y += 32

        if self.selected_mode != 0:
            y += 40
            diff_title = self.font.render("Select AI difficulty level:", True, TextColor)
            diff_title_rect = diff_title.get_rect(center=(SCREEN_WIDTH // 2, y))
            self.screen.blit(diff_title, diff_title_rect)
            y += 40
            for idx, text in enumerate(self.difficulty_options):
                color = HighlightColor if idx == self.selected_difficulty else TextColor
                option_text = self.font.render(text, True, color)
                rect = option_text.get_rect(center=(SCREEN_WIDTH // 2, y))  # <-- centrowanie
                self.screen.blit(option_text, rect)
                self.diff_rects.append(rect)
                y += 32

        start_text = self.font.render("START GAME", True, TextColor)
        self.start_rect = start_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 120))
        self.screen.blit(start_text, self.start_rect)

        theme_text = self.font.render("TOGGLE THEME", True, TextColor)
        self.theme_rect = theme_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 80))
        self.screen.blit(theme_text, self.theme_rect)

        close_text = self.font.render("CLOSE", True, TextColor)
        self.close_rect = close_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 40))
        self.screen.blit(close_text, self.close_rect)

        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while self.menu_active:
            self.draw_menu()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    # Plansze
                    for idx, rect in enumerate(self.board_rects):
                        if rect.collidepoint(pos):
                            self.selected_board = idx
                    # Tryby gry
                    for idx, rect in enumerate(self.mode_rects):
                        if rect.collidepoint(pos):
                            self.selected_mode = idx
                    # Poziom trudności
                    if self.selected_mode != 0:
                        for idx, rect in enumerate(self.diff_rects):
                            if rect.collidepoint(pos):
                                self.selected_difficulty = idx
                    # Start
                    if self.start_rect.collidepoint(pos):
                        self.menu_active = False
                    # Toggle theme
                    if self.theme_rect.collidepoint(pos):
                        self.toggle_theme()
                    # Close
                    if self.close_rect.collidepoint(pos):
                        pygame.quit()
                        sys.exit()

            clock.tick(FPS)
        return self.selected_board + 1, self.selected_mode, self.selected_difficulty

    def toggle_theme(self):
        global BgColor, TextColor, HighlightColor
        if BgColor == BLACK_TYPE[0]:
            BgColor, HighlightColor, TextColor = WHITE_TYPE[0], WHITE_TYPE[1], WHITE_TYPE[2]
        elif BgColor == WHITE_TYPE[0]:
            BgColor, HighlightColor, TextColor = RETRO_TYPE[0], RETRO_TYPE[1], RETRO_TYPE[2]
        elif BgColor == RETRO_TYPE[0]:
            BgColor, HighlightColor, TextColor = BLACK_TYPE[0], BLACK_TYPE[1], BLACK_TYPE[2]



class GameGUI:
    def __init__(self, board_choice, ai_mode, ai_difficulty):
        self.board_choice = board_choice
        self.mode_choice = ai_mode
        self.ai_difficulty = ai_difficulty


        # Ustal planszę na podstawie wyboru użytkownika:
        standard_boards = {
            1: ThreeMensMorrisBoard,
            2: SixMensMorrisBoard,
            3: NineMensMorrisBoard,
            4: TwelveMensMorrisBoard
        }
        self.board = standard_boards.get(board_choice, NineMensMorrisBoard)()
        self.state = self.board.get_initial_board_state()

        # Ustawienie graczy – zgodnie z wybranym trybem:
        self.players = [Player.WHITE, Player.BLACK]
        if ai_mode == 0:
            self.ai_player = Player.NONE  # Human vs Human
        elif ai_mode == 1:
            self.ai_player = Player.BLACK  # AI jako BLACK
        else:
            self.ai_player = Player.WHITE  # AI jako WHITE

        # Obsługa różnych modeli AI
        if self.ai_player != Player.NONE:
            if ai_difficulty in [0, 1, 2, 3]:  # Minimax
                self.ai = Minimax(self.board, ai_difficulty + 1)
            elif ai_difficulty == 4:  # Monte Carlo (0.1s)
                self.ai = MonteCarloTreeSearch(self.board)
                self.ai_time_limit = 0.1
            elif ai_difficulty == 5:  # Monte Carlo (1s)
                self.ai = MonteCarloTreeSearch(self.board)
                self.ai_time_limit = 1.0
            elif ai_difficulty == 6:  # Graph-based AI
                from engines.Unbitable import GraphAgent
                # Select the appropriate graph file based on the board type
                if isinstance(self.board, ThreeMensMorrisBoard):
                    graph_path = "states_graph/Threegame_graph_evaluated.txt"
                else:
                    # For other boards, use a default fallback (like Minimax)
                    self.ai = Minimax(self.board, 3)  # Fallback to Hard difficulty
                    print("Graph AI is only available for Three Men's Morris board")
                    self.ai_time_limit = 0
                    return
                    
                self.ai = GraphAgent(self.board, graph_path)
                self.ai_time_limit = 0  # Graph agent doesn't need time limit
        else:
            self.ai_time_limit = 0

        self.screen = pygame.display.get_surface()
        self.clock = pygame.time.Clock()
        self.coords = get_position_coordinates(self.board)
        self.selected_pos = None
        self.running = True

    def draw_board(self):
        self.screen.fill(BgColor)
        
        # Sprawdzenie czy mamy współrzędne
        if not self.coords:
            font = pygame.font.Font(FONT_PATH, 36)
            error_text = font.render("BŁĄD: Brak współrzędnych dla tej planszy!", True, (255, 0, 0))
            self.screen.blit(error_text, (50, 50))
            pygame.display.flip()
            return
        line_thikness = 4
        # Rysowanie linii łączących punkty planszy
        # Dla Nine Men's Morris
        if isinstance(self.board, NineMensMorrisBoard):
            for start,_, end in NINE_MEN_MILLS:
                pygame.draw.line(self.screen, TextColor, self.coords[start], self.coords[end], line_thikness)

        if isinstance(self.board, ThreeMensMorrisBoard):
            for start,_, end in THREE_MEN_MILLS:
                pygame.draw.line(self.screen, TextColor, self.coords[start], self.coords[end], line_thikness)
        if isinstance(self.board, TwelveMensMorrisBoard):
            for start,_, end in TWELVE_MEN_MILLS:
                pygame.draw.line(self.screen, TextColor, self.coords[start], self.coords[end], line_thikness)
        if isinstance(self.board, SixMensMorrisBoard):
            for start,_, end in SIX_MEN_MILLS:
                pygame.draw.line(self.screen, TextColor, self.coords[start], self.coords[end], line_thikness)
                pygame.draw.line(self.screen, TextColor, self.coords[1], self.coords[4], line_thikness)
                pygame.draw.line(self.screen, TextColor, self.coords[6], self.coords[7], line_thikness)
                pygame.draw.line(self.screen, TextColor, self.coords[8], self.coords[9], line_thikness)
                pygame.draw.line(self.screen, TextColor, self.coords[11], self.coords[14], line_thikness)
        
        # Rysowanie punktów planszy
        for pos_id, pos in self.coords.items():
            player = self.state.get_player_at_position(pos_id)
            if player == Player.WHITE:
                color = WHITE_COLOR
            elif player == Player.BLACK:
                color = BLACK_COLOR
            else:
                color = TextColor
            
            # Zaznaczenie wybranej pozycji
            if self.selected_pos == pos_id:
                pygame.draw.circle(self.screen, HighlightColor, pos, 23)
            if color == TextColor:
                pygame.draw.circle(self.screen, color, pos, 10)
            else:
                pygame.draw.circle(self.screen, color, pos, 16)
        # Dodanie informacji o aktualnym graczu
        font = pygame.font.Font(FONT_PATH, 40)
        current_player = "WHITE" if self.state.current_player == Player.WHITE else "BLACK"
        typ = " (AI Thinks)" if self.state.current_player == self.ai_player else ": remove" if self.state.need_to_remove_piece else ""
        player_text = font.render(f"PLAYER  {current_player}"+typ+" Board:" + str(self.board.board_state.to_int()), True, TextColor)
        player_text_rect = player_text.get_rect(topleft=(50, SCREEN_HEIGHT - 50))
        self.screen.blit(player_text, player_text_rect)
        
        pygame.display.flip()

    def handle_human_input(self):
        """
        Dla uproszczenia przyjmujemy, że ruchy typu PLACE są wykonywane przez pojedyncze kliknięcie.
        Dla ruchów MOVE – pierwsze kliknięcie wybiera pionek, drugie kliknięcie cel.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    action = show_pause_screen(self)
                    if action == "resume":
                        return
                    elif action == "play_again":
                        self.__init__(self.board_choice, self.mode_choice, self.ai_difficulty)
                        self.run()
                        return
                    elif action == "menu":
                        return "menu"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                clicked_id = None
                for pos_id, pos in self.coords.items():
                    dx = mouse_pos[0] - pos[0]
                    dy = mouse_pos[1] - pos[1]
                    if dx*dx + dy*dy <= 15*15:
                        clicked_id = pos_id
                        break
                if clicked_id is not None:
                    self.process_move_selection(clicked_id)

    def process_move_selection(self, clicked_id):
        legal_moves = self.board.get_legal_moves(self.state)
        # Uwzględniamy także ruchy REMOVE
        possible_moves = [
            m for m in legal_moves
            if (
                (m.move_type == MoveType.REMOVE and m.remove_checker_from_position.id == clicked_id)
                or (m.move_type == MoveType.PLACE and m.to_position.id == clicked_id)
                or (
                    m.move_type == MoveType.MOVE and 
                    (
                        (self.selected_pos is None and m.from_position.id == clicked_id) 
                        or (self.selected_pos is not None and m.to_position.id == clicked_id)
                    )
                )
            )
        ]
        if not possible_moves:
            self.selected_pos = None
            return

        # Najpierw obsługujemy ruchy REMOVE
        for m in possible_moves:
            if m.move_type == MoveType.REMOVE:
                self.state = self.board.make_move(self.state, m)
                self.selected_pos = None
                return

        # Jeśli mamy ruch PLACE:
        for m in possible_moves:
            if m.move_type == MoveType.PLACE:
                self.state = self.board.make_move(self.state, m)
                self.selected_pos = None
                return

        # Jeśli ruch MOVE:
        if self.selected_pos is None:
            piece = self.state.get_player_at_position(clicked_id)
            if piece == self.state.current_player:
                self.selected_pos = clicked_id
        else:
            for m in legal_moves:
                if m.move_type == MoveType.MOVE:
                    if m.from_position.id == self.selected_pos and m.to_position.id == clicked_id:
                        self.state = self.board.make_move(self.state, m)
                        self.selected_pos = None
                        return
            self.selected_pos = None

    def play_ai_turn(self):
        if isinstance(self.ai, MonteCarloTreeSearch):
            ai_move = self.ai.get_best_move(self.state, self.ai_time_limit)
        elif isinstance(self.ai, Minimax):
            ai_move = self.ai.get_best_move(self.state, self.ai_difficulty)
        else: ai_move = self.ai.get_best_move(self.state)
        if ai_move is None:
            self.running = False
            return
        self.state = self.board.make_move(self.state, ai_move)

    def run(self):
        while self.running:
            if self.board.check_if_game_is_over(self.state):
                winner = self.board.get_winner(self.state)
                # Zakończenie pętli i przejście do obsługi ekranu końcowego
                self.running = False
                break

            if self.state.current_player == self.ai_player:
                self.play_ai_turn()
            else:
                result = self.handle_human_input()
                if result == "menu":
                    return "menu"

            self.draw_board()
            self.clock.tick(FPS)

        winner = self.board.get_winner(self.state)
        action = show_end_screen(self, winner)

        if action == "play_again":
            self.__init__(self.board_choice, self.mode_choice, self.ai_difficulty)
            self.run()
            return
        elif action == "menu":
            # Przerywamy, by w main() wrócić do menu
            return "menu"
        else:
            pygame.quit()
            sys.exit()


def show_pause_screen(self):
    """
    Ekran pauzy wyświetlany po naciśnięciu ESC podczas rozgrywki,
    daje możliwość wyjścia do menu, zagrania ponownie lub powrotu do gry.
    """
    font = pygame.font.Font(FONT_PATH, 48)
    small_font = pygame.font.Font(FONT_PATH, 36)
    running = True

    while running:
        self.screen.fill(BgColor)

        pause_text = font.render("Pause", True, HighlightColor)
        self.screen.blit(pause_text, (SCREEN_WIDTH//2 - pause_text.get_width()//2, 200))

        tryagain_surface = small_font.render("PLAY AGAIN", True, TextColor)
        tryagain_rect = tryagain_surface.get_rect(center=(SCREEN_WIDTH//2, 350))
        pygame.draw.rect(self.screen, BgColor, tryagain_rect.inflate(20, 10), border_radius=10)
        self.screen.blit(tryagain_surface, tryagain_rect)

        exit_surface = small_font.render("EXIT TO MENU", True, TextColor)
        exit_rect = exit_surface.get_rect(center=(SCREEN_WIDTH//2, 420))
        pygame.draw.rect(self.screen, BgColor, exit_rect.inflate(20, 10), border_radius=10)
        self.screen.blit(exit_surface, exit_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "resume"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if tryagain_rect.collidepoint(event.pos):
                    return "play_again"
                elif exit_rect.collidepoint(event.pos):
                    return "menu"


def show_end_screen(self, winner):
    font = pygame.font.Font(FONT_PATH, 48)
    small_font = pygame.font.Font(FONT_PATH, 36)

    running = True
    while running:
        self.screen.fill(BgColor)

        win_text = f"Winner: {'WHITE' if winner == Player.WHITE else 'BLACK'}"
        win_surface = font.render(win_text, True, HighlightColor)
        self.screen.blit(win_surface, (SCREEN_WIDTH//2 - win_surface.get_width()//2, 200))
        tryagain_surface = small_font.render("PLAY AGAIN", True, TextColor)
        tryagain_rect = tryagain_surface.get_rect(center=(SCREEN_WIDTH//2, 350))
        pygame.draw.rect(self.screen, BgColor, tryagain_rect.inflate(20, 10), border_radius=10)
        self.screen.blit(tryagain_surface, tryagain_rect)

        exit_surface = small_font.render("EXIT TO MENU", True, TextColor)
        exit_rect = exit_surface.get_rect(center=(SCREEN_WIDTH//2, 420))
        pygame.draw.rect(self.screen, BgColor, exit_rect.inflate(20, 10), border_radius=10)
        self.screen.blit(exit_surface, exit_rect)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if tryagain_rect.collidepoint(event.pos):
                    return "play_again"
                elif exit_rect.collidepoint(event.pos):
                    return "menu"


# --- FUNKCJA GŁÓWNA ---
def main():
    global BgColor, TextColor, HighlightColor
    BgColor,HighlightColor, TextColor = BLACK_TYPE[0], BLACK_TYPE[1], BLACK_TYPE[2]
    while True:
        pygame.init()
        info = pygame.display.Info()  # Pobiera rozdzielczość ekranu
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)  # Ustawiamy rozmiar okna
        pygame.display.set_caption("Morris Game in Pygame")
        menu = GameMenu(screen)
        board_choice, mode_choice, ai_difficulty = menu.run()
        game_gui = GameGUI(board_choice, mode_choice, ai_difficulty)

        result = game_gui.run()
        # Jeśli "menu", wywołujemy pętlę jeszcze raz (wracamy do głównego menu)
        if result == "menu":
            continue
        else:
            break

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
