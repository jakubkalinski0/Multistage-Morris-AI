from Board import Board
from BoardState import BoardState
from Move import Move
from MoveType import MoveType
from Position import Position
from Player import Player
from NineMensMorrisBoard import NineMensMorrisBoard
from ThreeMensMorrisBoard import ThreeMensMorrisBoard
from SixMensMorrisBoard import SixMensMorrisBoard
from TwelveMensMorrisBoard import TwelveMensMorrisBoard
from Minimax import Minimax


class Game:
    def __init__(self):
        self.board = self._choose_board()()  # Instantiate the chosen board class
        self.state = self.board.get_initial_board_state()
        self.players = [Player.WHITE, Player.BLACK]
        self.ai_player = self._choose_ai_player()
        self.ai_difficulty = self._choose_ai_difficulty() if self.ai_player != Player.NONE else 0

        # Initialize AI if needed
        if self.ai_player != Player.NONE:
            self.ai = Minimax(self.board, self.ai_difficulty)

    def _choose_board(self):
        standard_available_boards = {
            1: ThreeMensMorrisBoard,
            2: SixMensMorrisBoard,
            3: NineMensMorrisBoard,
            4: TwelveMensMorrisBoard
        }

        while True:
            try:
                chosen_board = int(
                    input(f"You can choose one of the 4 following boards by typing in their index (1, 2, 3, 4):\n"
                          f"    [1] Three Men's Morris,\n"
                          f"    [2] Six Men's Morris,\n"
                          f"    [3] Nine Men's Morris,\n"
                          f"    [4] Twelve Men's Morris.\n").strip())

                if chosen_board in standard_available_boards:
                    return standard_available_boards[chosen_board]
                else:
                    print("Invalid choice. Please select a number between 1 and 4.")
            except ValueError:
                print("Please enter a valid number.")

    def _choose_ai_player(self):
        while True:
            try:
                print("Choose game mode:")
                print("[0] Human vs Human")
                print("[1] Human vs AI (you play as WHITE)")
                print("[2] AI vs Human (you play as BLACK)")
                choice = int(input("Enter your choice (0-2): ").strip())

                if choice == 0:
                    return Player.NONE  # No AI player
                elif choice == 1:
                    return Player.BLACK  # AI plays as BLACK
                elif choice == 2:
                    return Player.WHITE  # AI plays as WHITE
                else:
                    print("Invalid choice. Please select 0, 1, or 2.")
            except ValueError:
                print("Please enter a valid number.")

    def _choose_ai_difficulty(self):
        while True:
            try:
                print("Choose AI difficulty:")
                print("[1] Easy (1 ply)")
                print("[2] Medium (2 ply)")
                print("[3] Hard (3 ply)")
                print("[4] Expert (4 ply)")
                choice = int(input("Enter difficulty (1-4): ").strip())

                if 1 <= choice <= 4:
                    return choice
                else:
                    print("Invalid choice. Please select a number between 1 and 4.")
            except ValueError:
                print("Please enter a valid number.")

    def start(self):
        print("Game started!")
        self.board.print_map_2d()
        self.print_state_with_pieces()

        while not self.board.check_if_game_is_over(self.state):
            print(self.state)
            self.play_turn()

        self.end_game()

    def play_turn(self):
        current_player = self.state.current_player
        print(f"\n{current_player.name}'s turn")

        # Check if this is an AI player's turn
        if current_player == self.ai_player:
            self.play_ai_turn()
        else:
            self.play_human_turn()

    def play_human_turn(self):
        legal_moves = self.board.get_legal_moves(self.state)
        if not legal_moves:
            print(f"No legal moves available. Game over!")
            return

        # Display legal moves
        print("Legal moves:")
        for i, move in enumerate(legal_moves):
            print(f"[{i}] {self._move_to_string(move)}")

        # Get player's choice
        while True:
            try:
                choice = int(input("Enter the number of your move: "))
                if 0 <= choice < len(legal_moves):
                    selected_move = legal_moves[choice]
                    break
                else:
                    print(f"Invalid choice. Please select a number between 0 and {len(legal_moves) - 1}.")
            except ValueError:
                print("Please enter a valid number.")

        # Make the move
        self.state = self.board.make_move(self.state, selected_move)
        print(f"\nMove made: {self._move_to_string(selected_move)}")
        self.board.print_map_2d()
        self.print_state_with_pieces()

    def play_ai_turn(self):
        print("AI is thinking...")

        # Calculate time limit based on difficulty
        time_limit = min(1.0 + self.ai_difficulty, 5.0)  # 2-5 seconds based on difficulty

        # Get AI move
        ai_move = self.ai.get_best_move(self.state, time_limit)

        if not ai_move:
            print("AI could not find a valid move. Game over!")
            return

        # Make the move
        self.state = self.board.make_move(self.state, ai_move)
        print(f"\nAI move: {self._move_to_string(ai_move)}")
        self.board.print_map_2d()
        self.print_state_with_pieces()

    def _move_to_string(self, move: Move) -> str:
        """Convert a move to a human-readable string."""
        if move.move_type == MoveType.PLACE:
            return f"Place at position {move.to_position.id}"
        elif move.move_type == MoveType.MOVE:
            return f"Move from position {move.from_position.id} to position {move.to_position.id}"
        elif move.move_type == MoveType.REMOVE:
            return f"Remove piece at position {move.remove_checker_from_position.id}"
        return "Unknown move"

    def end_game(self):
        winner = self.board.get_winner(self.state)
        print("\nGame over!")
        if winner == Player.NONE:
            print("The game ended in a draw.")
        else:
            print(f"{winner.name} wins!")

        print("\nFinal board state:")
        print(self.state)
        self.board.print_map_2d()
        self.print_state_with_pieces()

    def print_state_with_pieces(self):
        """Print a representation of the board with pieces."""
        # Get the board representation as displayed by print_map_2d
        if isinstance(self.board, NineMensMorrisBoard):
            self._print_nine_mens_board_with_pieces()
        elif isinstance(self.board, ThreeMensMorrisBoard):
            self._print_three_mens_board_with_pieces()
        elif isinstance(self.board, SixMensMorrisBoard):
            self._print_six_mens_board_with_pieces()
        elif isinstance(self.board, TwelveMensMorrisBoard):
            self._print_twelve_mens_board_with_pieces()

    def _get_piece_symbol(self, position_id):
        player = self.state.get_player_at_position(position_id)
        if player == Player.WHITE:
            return "W"
        elif player == Player.BLACK:
            return "B"
        else:
            return "·"  # Empty position

    def _print_nine_mens_board_with_pieces(self):
        symbols = [self._get_piece_symbol(i) for i in range(24)]

        print("\nBoard with pieces:")
        print(f"{symbols[0]}----------{symbols[1]}---------{symbols[2]}")
        print("|          |         |")
        print(f"|    {symbols[3]}-----{symbols[4]}-----{symbols[5]}    |")
        print("|    |     |     |   |")
        print(f"|    |  {symbols[6]}--{symbols[7]}--{symbols[8]}  |   |")
        print("|    |  |     |  |   |")
        print(f"{symbols[9]}---{symbols[10]}-{symbols[11]}     {symbols[12]}-{symbols[13]}--{symbols[14]}")
        print("|    |  |     |  |   |")
        print(f"|    |  {symbols[15]}-{symbols[16]}-{symbols[17]}  |   |")
        print("|    |     |     |   |")
        print(f"|    {symbols[18]}----{symbols[19]}----{symbols[20]}   |")
        print("|          |         |")
        print(f"{symbols[21]}----------{symbols[22]}---------{symbols[23]}")
        print("Legend: W = White, B = Black, · = Empty")

    def _print_three_mens_board_with_pieces(self):
        symbols = [self._get_piece_symbol(i) for i in range(9)]

        print("\nBoard with pieces:")
        print(f"{symbols[0]}---{symbols[1]}---{symbols[2]}")
        print("| \\ | / |")
        print(f"{symbols[3]}---{symbols[4]}---{symbols[5]}")
        print("| / | \\ |")
        print(f"{symbols[6]}---{symbols[7]}---{symbols[8]}")
        print("Legend: W = White, B = Black, · = Empty")

    def _print_six_mens_board_with_pieces(self):
        symbols = [self._get_piece_symbol(i) for i in range(16)]

        print("\nBoard with pieces:")
        print(f"{symbols[0]}-------{symbols[1]}-------{symbols[2]}")
        print("|       |       |")
        print(f"|   {symbols[3]}---{symbols[4]}---{symbols[5]}   |")
        print("|   |       |   |")
        print(f"{symbols[6]}---{symbols[7]}       {symbols[8]}---{symbols[9]}")
        print("|   |       |   |")
        print(f"|   {symbols[10]}-{symbols[11]}-{symbols[12]}  |")
        print("|       |       |")
        print(f"{symbols[13]}------{symbols[14]}------{symbols[15]}")
        print("Legend: W = White, B = Black, · = Empty")

    def _print_twelve_mens_board_with_pieces(self):
        symbols = [self._get_piece_symbol(i) for i in range(24)]

        print("\nBoard with pieces:")
        print(f"{symbols[0]}---------{symbols[1]}---------{symbols[2]}")
        print("| \\       |       / |")
        print(f"|   {symbols[3]}-----{symbols[4]}-----{symbols[5]}   |")
        print("|   | \\   |   / |   |")
        print(f"|   |  {symbols[6]}--{symbols[7]}--{symbols[8]}  |   |")
        print("|   |  |     |  |   |")
        print(f"{symbols[9]}--{symbols[10]}-{symbols[11]}     {symbols[12]}-{symbols[13]}-{symbols[14]}")
        print("|   |  |     |  |   |")
        print(f"|   |  {symbols[15]}-{symbols[16]}-{symbols[17]} |   |")
        print("|   | /   |   \\ |   |")
        print(f"|   {symbols[18]}----{symbols[19]}----{symbols[20]}  |")
        print("| /       |       \\ |")
        print(f"{symbols[21]}--------{symbols[22]}--------{symbols[23]}")
        print("Legend: W = White, B = Black, · = Empty")


if __name__ == "__main__":
    game = Game()
    game.start()