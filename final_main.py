import time
import arcade
import sys
import math
import copy
import os
import random
from enum import Enum
from typing import List, Tuple, Optional

# --- Window Constants ---
GRID_ROWS = 5
GRID_COLS = 5
CELL_SIZE = 80
MARGIN = 80
WINDOW_WIDTH = GRID_COLS * CELL_SIZE + 2 * MARGIN
WINDOW_HEIGHT = GRID_ROWS * CELL_SIZE + 2 * MARGIN + 150
FPS = 60

# --- Color Definitions (Arcade uses RGB tuples as well) ---
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
LIGHT_GRAY = (220, 220, 220)
BLUE = (65, 105, 225)
DARK_BLUE = (25, 50, 150)
RED = (220, 60, 60)
DARK_RED = (150, 30, 30)
GREEN = (60, 200, 60)
YELLOW = (255, 215, 0)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)
PURPLE = (147, 112, 219)
ORANGE = (255, 140, 0)

# Gradient palettes (can be reused for dynamic drawing)
GRADIENT_BLUE = [(30, 60, 120), (65, 105, 225), (100, 149, 237)]
GRADIENT_RED = [(120, 30, 30), (220, 60, 60), (255, 100, 100)]
GRADIENT_GREEN = [(30, 120, 30), (60, 200, 60), (100, 255, 100)]

# --- ENUMS ---
class PieceType(Enum):
    KING = "King"
    QUEEN = "Queen"
    ROOK = "Rook"
    KNIGHT = "Knight"
    BISHOP = "Bishop"
    PAWN = "Pawn"

class Player(Enum):
    HUMAN = 1
    AI = 2
    PLAYER1 = 3
    PLAYER2 = 4

class GameMode(Enum):
    PVP = "Player vs Player"
    PVA = "Player vs AI"

# --- ANIMATION HANDLER ---
class Animation:
    def __init__(self, start_pos, end_pos, duration=0.5):
        """duration in seconds (Arcade uses float seconds, not ms)"""
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.duration = duration
        self.start_time = arcade.get_time()
        self.active = True

    def get_current_pos(self):
        current_time = arcade.get_time()
        elapsed = current_time - self.start_time

        if elapsed >= self.duration:
            self.active = False
            return self.end_pos

        progress = elapsed / self.duration
        # Smooth easing cubic
        progress = 1 - (1 - progress) ** 3

        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * progress
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * progress
        return (x, y)

# --- SIMPLE PARTICLE CLASS (Arcade-Compatible) ---
class Particle:
    def __init__(self, x, y, color, velocity, lifetime=1.0):
        self.x = x
        self.y = y
        self.color = color
        self.vx, self.vy = velocity
        self.lifetime = lifetime
        self.age = 0
        self.size = random.uniform(2, 5)

    def update(self, dt):
        """Update motion and check if still alive."""
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 300 * dt  # gravity in px/sec²
        self.age += dt
        return self.age < self.lifetime

    def draw(self):
        if self.age < self.lifetime:
            fade = 1 - (self.age / self.lifetime)
            size = max(1, self.size * fade)
            arcade.draw_circle_filled(self.x, self.y, size, self.color)

# --- CORE GAME STATE (Unchanged Logic) ---
class GameState:
    def __init__(self, rows=GRID_ROWS, cols=GRID_COLS):
        self.rows = rows
        self.cols = cols
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.player1_pos = None
        self.player2_pos = None
        self.current_player = Player.PLAYER1
        self.player1_piece = PieceType.QUEEN
        self.player2_piece = PieceType.QUEEN
        self.game_over = False
        self.winner = None

    def is_valid_position(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] == 0

    def get_valid_moves(self, player, piece_type, position):
        if not position:
            return []

        row, col = position
        moves = []

        if piece_type == PieceType.KING:
            directions = [(-1, -1), (-1, 0), (-1, 1),
                          (0, -1), (0, 1),
                          (1, -1), (1, 0), (1, 1)]
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if self.is_valid_position(new_row, new_col):
                    moves.append((new_row, new_col))

        elif piece_type == PieceType.QUEEN:
            directions = [(-1, -1), (-1, 0), (-1, 1),
                          (0, -1), (0, 1),
                          (1, -1), (1, 0), (1, 1)]
            for dr, dc in directions:
                for i in range(1, max(self.rows, self.cols)):
                    new_row, new_col = row + i * dr, col + i * dc
                    if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                        if self.grid[new_row][new_col] == 0:
                            moves.append((new_row, new_col))
                        else:
                            break
                    else:
                        break

        elif piece_type == PieceType.ROOK:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                for i in range(1, max(self.rows, self.cols)):
                    new_row, new_col = row + i * dr, col + i * dc
                    if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                        if self.grid[new_row][new_col] == 0:
                            moves.append((new_row, new_col))
                        else:
                            break
                    else:
                        break

        elif piece_type == PieceType.KNIGHT:
            knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                            (1, -2), (1, 2), (2, -1), (2, 1)]
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                if self.is_valid_position(new_row, new_col):
                    moves.append((new_row, new_col))

        elif piece_type == PieceType.BISHOP:
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in directions:
                for i in range(1, max(self.rows, self.cols)):
                    new_row, new_col = row + i * dr, col + i * dc
                    if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                        if self.grid[new_row][new_col] == 0:
                            moves.append((new_row, new_col))
                        else:
                            break
                    else:
                        break

        elif piece_type == PieceType.PAWN:
            direction = -1 if player in [Player.PLAYER1, Player.HUMAN] else 1
            new_row = row + direction
            if self.is_valid_position(new_row, col):
                moves.append((new_row, col))

        return moves

    def make_move(self, player, new_pos):
        if player in [Player.PLAYER1, Player.HUMAN] and self.player1_pos:
            old_row, old_col = self.player1_pos
            self.grid[old_row][old_col] = -1
            new_row, new_col = new_pos
            self.grid[new_row][new_col] = 1
            self.player1_pos = new_pos
        elif player in [Player.PLAYER2, Player.AI] and self.player2_pos:
            old_row, old_col = self.player2_pos
            self.grid[old_row][old_col] = -1
            new_row, new_col = new_pos
            self.grid[new_row][new_col] = 2
            self.player2_pos = new_pos

    def copy(self):
        new_state = GameState(self.rows, self.cols)
        new_state.grid = [row[:] for row in self.grid]
        new_state.player1_pos = self.player1_pos
        new_state.player2_pos = self.player2_pos
        new_state.current_player = self.current_player
        new_state.player1_piece = self.player1_piece
        new_state.player2_piece = self.player2_piece
        new_state.game_over = self.game_over
        new_state.winner = self.winner
        return new_state

class IsolationAI:
    def __init__(self, depth: int = 4):
        """AI for Isolation game using minimax with alpha-beta pruning."""
        self.depth = depth
        self.infinity = float('inf')

    # ------------------------------
    # Evaluation Function
    # ------------------------------
    def evaluate(self, state):
        ai_moves_now = len(state.get_valid_moves(Player.AI, state.player2_piece, state.player2_pos))
        human_moves_now = len(state.get_valid_moves(Player.HUMAN, state.player1_piece, state.player1_pos))

        # Terminal state checks
        if ai_moves_now == 0:
            return -1000
        if human_moves_now == 0:
            return 1000

        ai_future = self.best_future_mobility(state, Player.AI)
        human_future = self.best_future_mobility(state, Player.HUMAN)
        block_bonus = self.calculate_block_bonus(state)

        # Weighted mobility-based heuristic
        return (ai_moves_now - human_moves_now) + (ai_future - human_future) + block_bonus

    # ------------------------------
    # Heuristic Helpers
    # ------------------------------
    def best_future_mobility(self, state, player):
        """Estimate how many future moves a player might have."""
        best = 0
        piece = state.player2_piece if player == Player.AI else state.player1_piece
        pos = state.player2_pos if player == Player.AI else state.player1_pos

        if not pos:
            return 0

        for move in state.get_valid_moves(player, piece, pos):
            temp = state.copy()
            temp.make_move(player, move)
            mobility = len(temp.get_valid_moves(player, piece, move))
            best = max(best, mobility)

        return best

    def calculate_block_bonus(self, state):
        """Reward AI if it can move into human's most mobile square."""
        if not state.player1_pos or not state.player2_pos:
            return 0

        # Find human's best mobility move
        best_human_square = None
        best_human_mobility = -1
        for move in state.get_valid_moves(Player.HUMAN, state.player1_piece, state.player1_pos):
            temp = state.copy()
            temp.make_move(Player.HUMAN, move)
            mobility = len(temp.get_valid_moves(Player.HUMAN, state.player1_piece, move))
            if mobility > best_human_mobility:
                best_human_mobility = mobility
                best_human_square = move

        # If AI can occupy that square, give a bonus
        ai_moves = state.get_valid_moves(Player.AI, state.player2_piece, state.player2_pos)
        return 5 if best_human_square in ai_moves else 0

    # ------------------------------
    # Minimax Algorithm
    # ------------------------------
    def minimax(self, state, depth, alpha, beta, maximizing):
        """Recursive minimax with alpha-beta pruning."""
        if depth == 0 or self.is_terminal(state):
            return self.evaluate(state)

        if maximizing:
            max_eval = -self.infinity
            for move in state.get_valid_moves(Player.AI, state.player2_piece, state.player2_pos):
                temp = state.copy()
                temp.make_move(Player.AI, move)
                eval_score = self.minimax(temp, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = self.infinity
            for move in state.get_valid_moves(Player.HUMAN, state.player1_piece, state.player1_pos):
                temp = state.copy()
                temp.make_move(Player.HUMAN, move)
                eval_score = self.minimax(temp, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    # ------------------------------
    # Terminal Check
    # ------------------------------
    def is_terminal(self, state):
        ai_moves = state.get_valid_moves(Player.AI, state.player2_piece, state.player2_pos)
        human_moves = state.get_valid_moves(Player.HUMAN, state.player1_piece, state.player1_pos)
        return len(ai_moves) == 0 or len(human_moves) == 0

    # ------------------------------
    # Best Move Wrapper
    # ------------------------------
    def get_best_move(self, state):
        """Return the best move for AI based on evaluation."""
        best_score = -self.infinity
        best_move = None

        for move in state.get_valid_moves(Player.AI, state.player2_piece, state.player2_pos):
            temp = state.copy()
            temp.make_move(Player.AI, move)
            score = self.minimax(temp, self.depth - 1, -self.infinity, self.infinity, False)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

class IsolationGame(arcade.Window):
    def __init__(self):
        super().__init__(WINDOW_WIDTH, WINDOW_HEIGHT, "Enhanced Isolation Game")
        arcade.set_background_color(arcade.color.DARK_BLUE_GRAY)

        # ---------------------------
        # Core setup
        # ---------------------------
        self.last_time = 0.0
        #self.delta_time = 0.0
        self.frame_count = 0

        # Default grid
        self.grid_rows, self.grid_cols = 5, 5
        self.grid_input_text = ""
        self.active_input = False

        # Game state & AI
        self.state = GameState(self.grid_rows, self.grid_cols)
        self.ai = IsolationAI(depth=4)

        # Game control variables
        self.selected_pos = None
        self.valid_moves = []
        self.game_stage = "main_menu"  # main_menu, mode_selection, grid_selection, piece_selection, playing, game_over
        self.game_mode = None
        self.animations = []
        self.particles = []
        self.hover_pos = None
        self.button_animations = {}
        self.piece_selection_phase = "player1"
        self.bg_offset = 0

        # ---------------------------
        # Background
        # ---------------------------
        base_path = os.path.dirname(os.path.abspath(__file__))
        bg_path = os.path.join(base_path, "background.png")

        if os.path.exists(bg_path):
            self.background = arcade.load_texture(bg_path)
        else:
            print("⚠️ Warning: background.png not found.")
            self.background = None

        # ---------------------------
        # Piece images (menu + board)
        # ---------------------------
        menu_icon_size = (80, 80)
        board_icon_size = (CELL_SIZE - 20, CELL_SIZE - 20)
        self.piece_images = {
            PieceType.KING: {
                "menu": self._load_image("king.png", menu_icon_size),
                "board": self._load_image("king.png", board_icon_size),
            },
            PieceType.QUEEN: {
                "menu": self._load_image("queen.png", menu_icon_size),
                "board": self._load_image("queen.png", board_icon_size),
            },
            PieceType.ROOK: {
                "menu": self._load_image("rook.png", menu_icon_size),
                "board": self._load_image("rook.png", board_icon_size),
            },
            PieceType.KNIGHT: {
                "menu": self._load_image("knight.png", menu_icon_size),
                "board": self._load_image("knight.png", board_icon_size),
            },
            PieceType.BISHOP: {
                "menu": self._load_image("bishop.png", menu_icon_size),
                "board": self._load_image("bishop.png", board_icon_size),
            },
            PieceType.PAWN: {
                "menu": self._load_image("pawn.png", menu_icon_size),
                "board": self._load_image("pawn.png", board_icon_size),
            },
        }

        print("🎮 IsolationGame (Arcade) initialized successfully.")

    # ---------------------------
    # Utility: Load image safely
    # ---------------------------
    def _load_image(self, fname, size):
        """Helper to load and scale piece textures safely."""
        base_path = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base_path, fname)
        if not os.path.exists(path):
            print(f"⚠️ Warning: couldn't find {path}")
            return None

        try:
            # arcade automatically handles transparency (alpha channel)
            texture = arcade.load_texture(path)
            return texture
        except Exception as e:
            print(f"⚠️ Error loading {path}: {e}")
            return None

    def draw_grid_selection(self):
        """Draw screen for selecting grid size"""
        arcade.start_render()

        # Draw background
        if self.background:
            arcade.draw_lrwh_rectangle_textured(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, self.background)

        # Title
        arcade.draw_text(
            "Select Grid Size",
            WINDOW_WIDTH // 2,
            WINDOW_HEIGHT - 100,
            arcade.color.WHITE,
            48,
            anchor_x="center",
        )

        # Default grid button
        button_width, button_height = 300, 60
        default_y = WINDOW_HEIGHT - 220
        default_rect = arcade.Rect.from_kwargs(
            left=WINDOW_WIDTH // 2 - button_width / 2,
            bottom=default_y,
            width=button_width,
            height=button_height
        )
        self._draw_gradient_rect(
            WINDOW_WIDTH // 2 - button_width / 2,
            default_y,
            button_width,
            button_height,
            GRADIENT_BLUE
        )
        arcade.draw_rectangle_outline(
            default_rect.center_x, # Use rect center
            default_rect.center_y, # Use rect center
            button_width,
            button_height,
            arcade.color.WHITE,
            2
        )
        arcade.draw_text(
            "Use Default (5x5)",
            WINDOW_WIDTH // 2,
            default_y + button_height / 2 - 10,
            arcade.color.WHITE,
            24,
            anchor_x="center"
        )

        # Preset buttons (6x6 to 8x8)
        presets = [(6, 6), (7, 7), (8, 8)]
        buttons = []
        for i, (r, c) in enumerate(presets):
            y = default_y - (i + 1) * 80
            self._draw_gradient_rect(
                WINDOW_WIDTH // 2 - button_width // 2,
                y,
                button_width,
                button_height,
                GRADIENT_GREEN
            )
            center_x = self.width // 2
            center_y = default_y - (i + 1) * 80 + button_height / 2

            rect = arcade.Rect.from_kwargs(
                left=center_x - button_width / 2,
                bottom=center_y - button_height / 2,
                width=button_width,
                height=button_height
            )
            arcade.draw_rectangle_outline(
                center_x,
                center_y,
                button_width,
                button_height,
                arcade.color.WHITE,
                2
            )
            arcade.draw_text(
                f"{r} x {c}",
                WINDOW_WIDTH // 2,
                y + button_height / 2 - 10,
                arcade.color.WHITE,
                24,
                anchor_x="center"
            )
            buttons.append({
                "rect": rect,
                "rows": r,
                "cols": c
            })

        # Custom input box
        input_x, input_y, input_width, input_height = WINDOW_WIDTH // 2 - 150, 120, 300, 50

        # Create input box rectangle
        input_rect = arcade.Rect.from_kwargs(
            left=input_x,
            bottom=input_y,
            width=input_width,
            height=input_height
        )

        arcade.draw_lrbt_rectangle_outline(
            input_x,
            input_x + input_width,
            input_y + input_height,
            input_y,
            arcade.color.WHITE,
            2
        )

        if self.grid_input_text == "":
            text = "Type rows,cols (e.g. 6,7)"
            color = arcade.color.LIGHT_GRAY
        else:
            text = self.grid_input_text
            color = arcade.color.WHITE

        arcade.draw_text(
            text,
            input_x + 10,
            input_y + 15,
            color,
            18,
            anchor_x="left"
        )

        return {
            "default_rect": default_rect,
            "presets": buttons,
            "input_rect": input_rect
        }

    # ------------------------------------------------------------
    # Particle system (Arcade version)
    # ------------------------------------------------------------
    def create_particles(self, x, y, color, count=10):
        for _ in range(count):
            velocity = (random.uniform(-100, 100), random.uniform(-150, -50))
            particle = Particle(x, y, color, velocity)
            self.particles.append(particle)

    def update_particles(self, delta_time):
        self.particles = [p for p in self.particles if p.update(delta_time)]

    # ------------------------------------------------------------
    # Gradient rectangle (Arcade version)
    # ------------------------------------------------------------
    def _draw_gradient_rect(self, x, y, width, height, colors):
        """Draw a vertical gradient rectangle using Arcade shapes."""
        if len(colors) < 2:
            arcade.draw_lrbt_rectangle_filled(x, x + width, y + height, y, colors[0])
            return

        steps = height
        for i in range(int(steps)):
            ratio = i / steps
            color_index = ratio * (len(colors) - 1)
            lower_index = int(color_index)
            upper_index = min(lower_index + 1, len(colors) - 1)

            local_ratio = color_index - lower_index
            color1 = colors[lower_index]
            color2 = colors[upper_index]
            blended_color = tuple(int(c1 + (c2 - c1) * local_ratio) for c1, c2 in zip(color1, color2))

            arcade.draw_lrbt_rectangle_filled(
                x, x + width,
                y + height - i,
                y + height - i - 1,
                blended_color
            )

    # ------------------------------------------------------------
    # Draw piece icons (Arcade version)
    # ------------------------------------------------------------
    def draw_piece_icon(self, piece_type, x, y, size=30, color=arcade.color.BLACK):
        """Draw simplified chess-like icons using Arcade primitives"""
        cx, cy = x, y

        if piece_type == PieceType.KING:
            points = [
                (cx, cy + size // 2),
                (cx - size // 3, cy + size // 4),
                (cx - size // 2, cy),
                (cx - size // 4, cy - size // 4),
                (cx + size // 4, cy - size // 4),
                (cx + size // 2, cy),
                (cx + size // 3, cy + size // 4)
            ]
            arcade.draw_polygon_filled(points, color)
            arcade.draw_circle_filled(cx, cy + size // 4, 3, arcade.color.GOLD)

        elif piece_type == PieceType.QUEEN:
            points = [
                (cx, cy + size // 2),
                (cx - size // 4, cy + size // 3),
                (cx - size // 2, cy + size // 6),
                (cx - size // 3, cy),
                (cx, cy + size // 8),
                (cx + size // 3, cy),
                (cx + size // 2, cy + size // 6),
                (cx + size // 4, cy + size // 3)
            ]
            arcade.draw_polygon_filled(points, color)
            arcade.draw_lrbt_rectangle_filled(cx - size // 2, cx + size // 2, cy - size // 6, cy - size // 4, color)

        elif piece_type == PieceType.ROOK:
            arcade.draw_lrbt_rectangle_filled(cx - size // 2, cx + size // 2, cy + size // 3, cy - size // 3, color)
            for i in range(3):
                x_pos = cx - size // 3 + i * size // 3
                arcade.draw_lrbt_rectangle_filled(x_pos, x_pos + size // 6, cy + size // 2, cy + size // 3, color)
            arcade.draw_lrbt_rectangle_filled(cx - size // 3, cx + size // 3, cy - size // 2, cy - size // 3, color)

        elif piece_type == PieceType.KNIGHT:
            points = [
                (cx - size // 4, cy - size // 3),
                (cx - size // 2, cy),
                (cx - size // 3, cy + size // 2),
                (cx, cy + size // 3),
                (cx + size // 4, cy + size // 4),
                (cx + size // 3, cy),
                (cx + size // 4, cy - size // 3)
            ]
            arcade.draw_polygon_filled(points, color)
            arcade.draw_circle_filled(cx - size // 6, cy + size // 6, 2, arcade.color.WHITE)

        elif piece_type == PieceType.BISHOP:
            points = [
                (cx, cy + size // 2),
                (cx - size // 4, cy + size // 6),
                (cx - size // 3, cy - size // 6),
                (cx + size // 3, cy - size // 6),
                (cx + size // 4, cy + size // 6)
            ]
            arcade.draw_polygon_filled(points, color)
            arcade.draw_line(cx, cy + size // 2, cx, cy + size // 3, arcade.color.WHITE, 2)
            arcade.draw_line(cx - size // 8, cy + 3 * size // 8, cx + size // 8, cy + 3 * size // 8, arcade.color.WHITE, 2)

        elif piece_type == PieceType.PAWN:
            arcade.draw_circle_filled(cx, cy + size // 4, size // 4, color)
            arcade.draw_lrbt_rectangle_filled(cx - size // 6, cx + size // 6, cy, cy - size // 3, color)
            arcade.draw_lrbt_rectangle_filled(cx - size // 4, cx + size // 4, cy - size // 2, cy - size // 3, color)

    def draw_animated_background(self):
        """Draw animated background"""
        self.bg_offset += 0.5
        if self.bg_offset > 50:
            self.bg_offset = 0

        # Gradient background (simulate gradient with rectangles)
        arcade.draw_lrbt_rectangle_filled(0, self.width, self.height, 0, arcade.color.DARK_MIDNIGHT_BLUE)
        arcade.draw_lrbt_rectangle_filled(0, self.width, self.height * 0.7, 0, arcade.color.DARK_BLUE_GRAY)
        arcade.draw_lrbt_rectangle_filled(0, self.width, self.height * 0.4, 0, arcade.color.DARK_CERULEAN)

        # Moving pattern
        tick = (time.time() - self.start_time) * 1000
        for i in range(0, self.width + 100, 100):
            for j in range(0, self.height + 100, 100):
                x = i - self.bg_offset
                y = j - self.bg_offset * 0.5
                alpha = 30 + 20 * math.sin((x + y) * 0.01 + tick * 0.001)
                color_val = int(max(0, min(255, alpha * 1.5)))
                arcade.draw_circle_filled(x, y, 2, (color_val, color_val, color_val))

    def draw_main_menu(self):
        """Draw enhanced main menu (fixed layout)"""
        self.clear()

        # Draw background properly (fills whole window)
        if self.background:
            arcade.draw_texture_rect(
            self.background,
            arcade.rect.XYWH(self.width // 2, self.height // 2, self.width, self.height)
            )
        else:
            arcade.draw_lrbt_rectangle_filled(0, self.width, self.height, 0, arcade.color.DARK_BLUE_GRAY)

        # Main title (centered and visible)
        arcade.draw_text(
            "ISOLATION AI GAME",
            self.width // 2,
            self.height - 80,
            arcade.color.WHITE,
            45,
            anchor_x="center",
            anchor_y="center",
            font_name="Kenney Mini Square",
        )

        # Subtitle (smaller and lower)
        arcade.draw_text(
            "Strategic Board Game",
            self.width // 2,
            self.height - 220,
            arcade.color.LIGHT_GRAY,
            32,
            anchor_x="center",
            anchor_y="center",
        )

        # Animated "Press any key" hint
        alpha = int(127 + 127 * math.sin(time.time() * 3))
        arcade.draw_text(
            "Press any key to continue",
            self.width // 2,
            100,
            (255, 255, 255, alpha),
            24,
            anchor_x="center",
            anchor_y="center",
        )

    def draw_mode_selection(self):
        """Draw game mode selection screen"""
        if self.background:
            arcade.draw_texture_rect(
            self.background,
            arcade.rect.XYWH(self.width // 2, self.height // 2, self.width, self.height)
            )
        else:
            arcade.draw_lrbt_rectangle_filled(0, self.width, self.height, 0, arcade.color.DARK_BLUE_GRAY)

        arcade.draw_text("Select Game Mode", self.width // 2, self.height - 100,
                         arcade.color.WHITE, 48, anchor_x="center", anchor_y="center")

        button_width, button_height = 300, 80
        buttons = [
            {"text": "Player vs AI", "mode": GameMode.PVA, "y": 300, "color": arcade.color.ROYAL_BLUE},
            {"text": "Player vs Player", "mode": GameMode.PVP, "y": 180, "color": arcade.color.RED_DEVIL},
        ]

        mouse_x, mouse_y = self._get_mouse_position()

        for button in buttons:
            x = self.width // 2 - button_width // 2
            y = button["y"]
            center_x = self.width // 2
            center_y = y + button_height / 2

            # Create rectangle using lbwh helper method
            button["rect"] = arcade.Rect.from_kwargs(
                left=center_x - button_width / 2,
                bottom=center_y - button_height / 2,
                width=button_width,
                height=button_height
            )
            hovered = (x < mouse_x < x + button_width and y < mouse_y < y + button_height)

            color = button["color"]
            if hovered:
                # Draw with larger size for hover effect
                arcade.draw_lrbt_rectangle_filled(
                    center_x - button_width / 2 - 10,
                    center_x + button_width / 2 + 10,
                    center_y + button_height / 2 + 10,
                    center_y - button_height / 2 - 10,
                    color
                )
            else:
                # Draw with normal size
                arcade.draw_lrbt_rectangle_filled(
                    center_x - button_width / 2,
                    center_x + button_width / 2,
                    center_y + button_height / 2,
                    center_y - button_height / 2,
                    color
                )

            # Border and text
            arcade.draw_lrbt_rectangle_outline(
                center_x - button_width / 2,
                center_x + button_width / 2,
                center_y + button_height / 2,
                center_y - button_height / 2,
                arcade.color.WHITE,
                3
            )
            arcade.draw_text(button["text"], self.width // 2, y + button_height / 2,
                             arcade.color.WHITE, 24, anchor_x="center", anchor_y="center")

        arcade.draw_text("Press ESC to go back", 20, 30, arcade.color.LIGHT_GRAY, 16)
        return buttons

    def draw_piece_selection(self):
        """Draw enhanced piece selection screen (Arcade version)"""
        arcade.draw_lrwh_rectangle_textured(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT, self.background)

        # Determine title and color
        if self.game_mode == GameMode.PVP:
            if self.piece_selection_phase == "player1":
                title = "Player 1: Select Your Piece"
                color = arcade.color.BLUE
            else:
                title = "Player 2: Select Your Piece"
                color = arcade.color.RED
        else:
            title = "Select Your Piece"
            color = arcade.color.BLUE

        arcade.draw_text(title, WINDOW_WIDTH // 2, WINDOW_HEIGHT - 80, color, 36, anchor_x="center")

        # Piece selection grid
        pieces = list(PieceType)
        cols, rows = 3, 2
        start_x = WINDOW_WIDTH // 2 - (cols * 120) // 2
        start_y = 300
        buttons = []

        for i, piece in enumerate(pieces):
            row = i // cols
            col = i % cols
            x = start_x + col * 120
            y = start_y - row * 100
            width, height = 100, 80

            # Create rectangle using lbwh helper method
            rect = arcade.Rect.from_kwargs(
                left=x,
                bottom=y,
                width=width,
                height=height
            )
            # ----------------

            # Hover detection using mouse coords
            mouse_x, mouse_y = self._get_mouse_position()
            hovered = x < mouse_x < x + width and y < mouse_y < y + height

            # Hover glow
            if hovered:
                arcade.draw_lrbt_rectangle_filled(x - 5, x + width + 5, y + height + 5, y - 5, arcade.color.AO)
                if random.random() < 0.3:
                    self.create_particles(x + width / 2, y + height / 2, arcade.color.GREEN, 2)

            # Button background
            arcade.draw_lrbt_rectangle_filled(x, x + width, y + height, y, arcade.color.GRAY_BLUE)
            arcade.draw_lrbt_rectangle_outline(x, x + width, y + height, y, arcade.color.WHITE, 2)

            # Draw piece image
            piece_img = self.piece_images[piece]["menu"]
            if piece_img:
                piece_img.draw_scaled(x + width / 2, y + height / 2 + 10, 0.5)

            # Label
            arcade.draw_text(piece.value, x + width / 2, y + 10, arcade.color.WHITE, 14, anchor_x="center")

            # --- UPDATED CODE ---
            buttons.append({"x": x, "y": y, "width": width, "height": height, "piece": piece, "rect": rect})
            # --------------------

        return buttons

    def draw_game_board(self):
        """Draw enhanced game board (Arcade version)"""
        # Board shadow
        arcade.draw_lrbt_rectangle_filled(
            MARGIN - 5,
            MARGIN + self.state.cols * CELL_SIZE + 5,
            MARGIN + self.state.rows * CELL_SIZE + 5,
            MARGIN - 5,
            arcade.color.DARK_SLATE_GRAY
        )

        # Grid cells
        for row in range(self.state.rows):
            for col in range(self.state.cols):
                x = MARGIN + col * CELL_SIZE
                y = MARGIN + row * CELL_SIZE
                color = arcade.color.WHITE

                if self.state.grid[row][col] == -1:
                    color = arcade.color.DARK_GRAY
                    arcade.draw_lrbt_rectangle_filled(x, x + CELL_SIZE, y + CELL_SIZE, y, color)
                    arcade.draw_line(x + 10, y + 10, x + CELL_SIZE - 10, y + CELL_SIZE - 10, arcade.color.RED, 3)
                    arcade.draw_line(x + CELL_SIZE - 10, y + 10, x + 10, y + CELL_SIZE - 10, arcade.color.RED, 3)

                elif (row, col) == self.selected_pos:
                    arcade.draw_lrbt_rectangle_filled(x, x + CELL_SIZE, y + CELL_SIZE, y, arcade.color.LIGHT_GREEN)
                elif (row, col) in self.valid_moves:
                    arcade.draw_lrbt_rectangle_filled(x, x + CELL_SIZE, y + CELL_SIZE, y, arcade.color.LIGHT_SAGE)
                else:
                    arcade.draw_lrbt_rectangle_filled(x, x + CELL_SIZE, y + CELL_SIZE, y, arcade.color.WHITE_SMOKE)

                arcade.draw_lrbt_rectangle_outline(x, x + CELL_SIZE, y + CELL_SIZE, y, arcade.color.BLACK, 2)

        self.draw_pieces()

        # Draw particles
        for particle in self.particles:
            particle.draw()

    def draw_pieces(self):
        """Draw player pieces with color gradients (Arcade version)"""
        piece_size = CELL_SIZE // 2 - 10 # Define a reasonable icon size

        # Player 1
        if self.state.player1_pos:
            row, col = self.state.player1_pos
            x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
            y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2

            arcade.draw_circle_filled(x, y, CELL_SIZE // 2 - 5, arcade.color.BLUE)
            arcade.draw_circle_outline(x, y, CELL_SIZE // 2 - 5, arcade.color.WHITE, 3)

            # --- UPDATED CODE ---
            self.draw_piece_icon(self.state.player1_piece, x, y, size=piece_size, color=arcade.color.WHITE)
            # --------------------
            text = "P1" if self.game_mode == GameMode.PVP else "YOU"
            arcade.draw_text(text, x, y - 35, arcade.color.WHITE, 14, anchor_x="center")

        # Player 2
        if self.state.player2_pos:
            row, col = self.state.player2_pos
            x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
            y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2

            arcade.draw_circle_filled(x, y, CELL_SIZE // 2 - 5, arcade.color.RED)
            arcade.draw_circle_outline(x, y, CELL_SIZE // 2 - 5, arcade.color.WHITE, 3)

            # --- UPDATED CODE ---
            self.draw_piece_icon(self.state.player2_piece, x, y, size=piece_size, color=arcade.color.WHITE)
            # --------------------
            text = "P2" if self.game_mode == GameMode.PVP else "AI"
            arcade.draw_text(text, x, y - 35, arcade.color.WHITE, 14, anchor_x="center")

    def draw_game_ui(self):
        """Draw enhanced game UI in Arcade"""
        # Top panel with gradient background
        arcade.draw_lrwh_rectangle_gradient(0, WINDOW_HEIGHT - MARGIN + 10, WINDOW_WIDTH, MARGIN - 10,
                                            (60, 80, 120), (40, 60, 100))

        # Current player indicator with glow effect
        if self.state.current_player == Player.PLAYER1:
            player_text = "Your Turn" if self.game_mode == GameMode.PVA else "Player 1's Turn"
            color = BLUE
        else:
            player_text = "AI Turn" if self.game_mode == GameMode.PVA else "Player 2's Turn"
            color = RED

        # Draw glowing text effect
        for offset in [(2, 2), (1, 1), (0, 0)]:
            text_color = DARK_GRAY if offset != (0, 0) else WHITE
            arcade.draw_text(player_text, 20 + offset[0], WINDOW_HEIGHT - MARGIN + 15 + offset[1],
                            text_color, 18, font_name="Arial", bold=True)

        # Move counter and game info
        move_info = f"Moves: P1({len(self.state.get_valid_moves(Player.PLAYER1, self.state.player1_piece, self.state.player1_pos))}) "
        move_info += f"P2({len(self.state.get_valid_moves(Player.PLAYER2, self.state.player2_piece, self.state.player2_pos))})"
        arcade.draw_text(move_info, WINDOW_WIDTH - 250, WINDOW_HEIGHT - MARGIN + 15, WHITE, 14)

        # Piece information
        piece_info = f"Pieces: {self.state.player1_piece.value} vs {self.state.player2_piece.value}"
        arcade.draw_text(piece_info, 20, WINDOW_HEIGHT - MARGIN + 40, LIGHT_GRAY, 14)

        # Bottom panel
        arcade.draw_lrwh_rectangle_gradient(0, 0, WINDOW_WIDTH, 60, (40, 60, 100), (60, 80, 120))

        # Instructions
        instruction = "Click a highlighted square to move" if self.selected_pos else "Click your piece to select it"
        color = GREEN if self.selected_pos else WHITE
        arcade.draw_text(instruction, 20, 20, color, 14)

        # Controls
        arcade.draw_text("ESC: Menu | R: Restart", WINDOW_WIDTH - 200, 20, LIGHT_GRAY, 14)

    def draw_game_over(self):
        """Draw enhanced game over screen in Arcade"""
        # Semi-transparent overlay
        arcade.draw_lrbt_rectangle_filled(0, WINDOW_WIDTH, WINDOW_HEIGHT, 0, (0, 0, 0, 180))

        # Winner announcement with effects
        if self.state.winner == Player.PLAYER1:
            winner_text = "You Win!" if self.game_mode == GameMode.PVA else "Player 1 Wins!"
            color = GRADIENT_BLUE
        else:
            winner_text = "AI Wins!" if self.game_mode == GameMode.PVA else "Player 2 Wins!"
            color = GRADIENT_RED

        # Animated winner text
        scale = 1 + 0.1 * math.sin(time.time() * 5)
        font_size = int(72 * scale)
        for offset in [(4, 4), (2, 2), (0, 0)]:
            text_color = color[0] if offset != (0, 0) else WHITE
            arcade.draw_text(winner_text, WINDOW_WIDTH // 2 + offset[0], WINDOW_HEIGHT // 2 - 50 + offset[1],
                            text_color, font_size, anchor_x="center", anchor_y="center", bold=True)

        # Game stats
        stats = [
            f"Final Moves: Player 1 had {len(self.state.get_valid_moves(Player.PLAYER1, self.state.player1_piece, self.state.player1_pos))} moves",
            f"Final Moves: Player 2 had {len(self.state.get_valid_moves(Player.PLAYER2, self.state.player2_piece, self.state.player2_pos))} moves",
            f"Pieces Used: {self.state.player1_piece.value} vs {self.state.player2_piece.value}"
        ]
        for i, stat in enumerate(stats):
            arcade.draw_text(stat, WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 20 + i * 25,
                            WHITE, 16, anchor_x="center")

        # Restart instructions
        arcade.draw_text("Press R to Restart or ESC for Menu", WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 120,
                        YELLOW, 18, anchor_x="center")

    def setup_game(self):
        """Initialize game positions for Arcade"""
        # Place player 1 at bottom-left
        self.state.player1_pos = (self.state.rows - 1, 0)
        self.state.grid[self.state.rows - 1][0] = 1

        # Place player 2 at top-right
        self.state.player2_pos = (0, self.state.cols - 1)
        self.state.grid[0][self.state.cols - 1] = 2

        # Create entrance particles
        self.create_particles(MARGIN + CELL_SIZE // 2,
                            MARGIN + (self.state.rows - 1) * CELL_SIZE + CELL_SIZE // 2,
                            BLUE, 15)
        self.create_particles(MARGIN + (self.state.cols - 1) * CELL_SIZE + CELL_SIZE // 2,
                            MARGIN + CELL_SIZE // 2,
                            RED, 15)

        self.game_stage = "playing"

    def handle_click(self, x, y):
        """Handle mouse clicks in Arcade"""
        if self.game_stage == "mode_selection":
            buttons = self.draw_mode_selection()
            for button in buttons:
                # --- UPDATED CODE ---
                if button["rect"].collides_with_point((x, y)):
                    self.game_mode = button["mode"]
                    self.game_stage = "grid_selection"
                    self.create_particles(x, y, WHITE, 8)
                    return

        elif self.game_stage == "piece_selection":
            buttons = self.draw_piece_selection()
            for button in buttons:
                # --- UPDATED CODE ---
                if button["rect"].collides_with_point((x, y)):
                    if self.game_mode == GameMode.PVP and self.piece_selection_phase == "player1":
                        self.state.player1_piece = button["piece"]
                        self.piece_selection_phase = "player2"
                        self.create_particles(button["rect"].center_x, button["rect"].center_y, BLUE, 10)
                    elif self.game_mode == GameMode.PVP and self.piece_selection_phase == "player2":
                        self.state.player2_piece = button["piece"]
                        self.setup_game()
                        self.create_particles(button["rect"].center_x, button["rect"].center_y, RED, 10)
                    else:  # PvA
                        self.state.player1_piece = button["piece"]
                        # The AI will use the same piece type as the human player in PVA
                        self.state.player2_piece = button["piece"]
                        self.setup_game()
                        self.create_particles(button["rect"].center_x, button["rect"].center_y, GREEN, 10)
                    return
                # --------------------

        elif self.game_stage == "grid_selection":
            buttons = self.draw_grid_selection()

            # --- UPDATED CODE ---
            if buttons["input_rect"].collides_with_point((x, y)):
                self.active_input = True
                return
            else:
                self.active_input = False

            if buttons["default_rect"].collides_with_point((x, y)):
                self.grid_rows, self.grid_cols = 5, 5
                self.state = GameState(self.grid_rows, self.grid_cols)
                self.resize_window()
                self.game_stage = "piece_selection"
                return

            for b in buttons["presets"]:
                if b["rect"].collides_with_point((x, y)):
                    self.grid_rows, self.grid_cols = b["rows"], b["cols"]
                    self.state = GameState(self.grid_rows, self.grid_cols)
                    self.resize_window()
                    self.game_stage = "piece_selection"
                    return
            # --------------------

        elif self.game_stage == "playing":
            col = (x - MARGIN) // CELL_SIZE
            row = (y - MARGIN) // CELL_SIZE
            if 0 <= row < self.state.rows and 0 <= col < self.state.cols:
                if self.state.current_player == Player.PLAYER1:
                    if (row, col) == self.state.player1_pos:
                        self.selected_pos = (row, col)
                        self.valid_moves = self.state.get_valid_moves(Player.PLAYER1, self.state.player1_piece, self.state.player1_pos)
                        self.create_particles(x, y, BLUE, 5)
                    elif (row, col) in self.valid_moves:
                        self.state.make_move(Player.PLAYER1, (row, col))
                        self.create_particles(x, y, GREEN, 12)
                        self.selected_pos = None
                        self.valid_moves = []
                        if self.check_game_over():
                            return
                        self.state.current_player = Player.PLAYER2
                    else:
                        self.selected_pos = None
                        self.valid_moves = []

                elif self.state.current_player == Player.PLAYER2 and self.game_mode == GameMode.PVP:
                    if (row, col) == self.state.player2_pos:
                        self.selected_pos = (row, col)
                        self.valid_moves = self.state.get_valid_moves(Player.PLAYER2, self.state.player2_piece, self.state.player2_pos)
                        self.create_particles(x, y, RED, 5)
                    elif (row, col) in self.valid_moves:
                        self.state.make_move(Player.PLAYER2, (row, col))
                        self.create_particles(x, y, GREEN, 12)
                        self.selected_pos = None
                        self.valid_moves = []
                        if self.check_game_over():
                            return
                        self.state.current_player = Player.PLAYER1
                    else:
                        self.selected_pos = None
                        self.valid_moves = []

    def ai_move(self):
        """Execute AI move in Arcade"""
        if self.state.current_player == Player.PLAYER2 and self.game_mode == GameMode.PVA:
            ai_state = self.state.copy()
            ai_state.current_player = Player.AI
            ai_state.human_pos = self.state.player1_pos
            ai_state.ai_pos = self.state.player2_pos
            ai_state.human_piece = self.state.player1_piece
            ai_state.ai_piece = self.state.player2_piece

            best_move = self.ai.get_best_move(ai_state)
            if best_move:
                row, col = best_move
                x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
                y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2
                self.create_particles(x, y, RED, 15)
                self.state.make_move(Player.PLAYER2, best_move)
                if self.check_game_over():
                    return
                self.state.current_player = Player.PLAYER1

    def check_game_over(self):
        """Check game over in Arcade"""
        player1_moves = self.state.get_valid_moves(Player.PLAYER1, self.state.player1_piece, self.state.player1_pos)
        player2_moves = self.state.get_valid_moves(Player.PLAYER2, self.state.player2_piece, self.state.player2_pos)

        if len(player1_moves) == 0:
            self.state.game_over = True
            self.state.winner = Player.PLAYER2
            self.game_stage = "game_over"
            for _ in range(30):
                x = random.randint(100, WINDOW_WIDTH - 100)
                y = random.randint(100, WINDOW_HEIGHT - 100)
                self.create_particles(x, y, RED, 3)
            return True
        elif len(player2_moves) == 0:
            self.state.game_over = True
            self.state.winner = Player.PLAYER1
            self.game_stage = "game_over"
            for _ in range(30):
                x = random.randint(100, WINDOW_WIDTH - 100)
                y = random.randint(100, WINDOW_HEIGHT - 100)
                self.create_particles(x, y, BLUE, 3)
            return True
        return False

    def resize_window(self):
        global WINDOW_WIDTH, WINDOW_HEIGHT
        WINDOW_WIDTH = self.grid_cols * CELL_SIZE + 2 * MARGIN
        WINDOW_HEIGHT = self.grid_rows * CELL_SIZE + 2 * MARGIN + 150
        arcade.set_window_size(WINDOW_WIDTH, WINDOW_HEIGHT)
        base_path = os.path.dirname(os.path.abspath(__file__))
        bg_path = os.path.join(base_path, "background.png")

        if os.path.exists(bg_path):
            self.background = arcade.load_texture(bg_path)
        else:
            print("⚠️ Warning: background.png not found after resizing.")
            self.background = None

    def reset_game(self):
        """Reset game to initial state"""
        self.state = GameState(self.grid_rows, self.grid_cols)
        self.selected_pos = None
        self.valid_moves = []
        self.game_stage = "mode_selection"
        self.piece_selection_phase = "player1"
        self.particles.clear()
        self.animations.clear()
        self.grid_input_text = ""

    def on_update(self, delta_time: float):
        """Arcade update function (replaces Pygame main loop)"""
        # Update particles and animations
        self.update_particles(delta_time)

        # AI move logic
        if (self.game_stage == "playing" and
            self.state.current_player == Player.PLAYER2 and
            self.game_mode == GameMode.PVA):
            self.ai_move()

    def on_draw(self):
        """Arcade draw function"""
        self.clear()

        # Draw game stage
        if self.game_stage == "main_menu":
            self.draw_main_menu()
        elif self.game_stage == "mode_selection":
            self.draw_mode_selection()
        elif self.game_stage == "grid_selection":
            self.draw_grid_selection()
        elif self.game_stage == "piece_selection":
            self.draw_piece_selection()
        elif self.game_stage == "playing":
            self.draw_animated_background()
            self.draw_game_board()
            self.draw_game_ui()
        elif self.game_stage == "game_over":
            self.draw_animated_background()
            self.draw_game_board()
            self.draw_game_over()

    def on_mouse_press(self, x, y, button, modifiers):
        """Arcade mouse click handler"""
        if button == arcade.MOUSE_BUTTON_LEFT:
            self.handle_click(x, y)

    def on_key_press(self, key, modifiers):
        """Arcade keyboard handler"""
        if key == arcade.key.ESCAPE:
            if self.game_stage in ["mode_selection", "grid_selection", "piece_selection"]:
                if self.game_stage == "piece_selection" and self.piece_selection_phase == "player2":
                    self.piece_selection_phase = "player1"
                else:
                    self.game_stage = "main_menu"
            else:
                self.reset_game()

        elif key == arcade.key.R and self.game_stage in ["playing", "game_over"]:
            self.reset_game()

        elif self.game_stage == "main_menu":
            self.game_stage = "mode_selection"

        elif self.game_stage == "grid_selection" and self.active_input:
            if key == arcade.key.BACKSPACE:
                self.grid_input_text = self.grid_input_text[:-1]
            elif key == arcade.key.ENTER:
                try:
                    r, c = map(int, self.grid_input_text.split(","))
                    if 5 <= r <= 8 and 5 <= c <= 8:
                        self.grid_rows, self.grid_cols = r, c
                        self.state = GameState(r, c)
                        self.resize_window()
                        self.game_stage = "piece_selection"
                    else:
                        print("Invalid grid size, must be between 5x5 and 8x8.")
                except Exception:
                    print("Invalid input format, use rows,cols (e.g., 6,7)")
            else:
                if chr(key).isdigit() or chr(key) == ",":
                    self.grid_input_text += chr(key)

    def _get_mouse_position(self):
        return getattr(self, "mouse_x", 0), getattr(self, "mouse_y", 0)

    def on_mouse_motion(self, x, y, dx, dy):
        self.mouse_x, self.mouse_y = x, y

    def run_arcade(self):
        """Run the game using Arcade"""
        arcade.run()

if __name__ == "__main__":
    game = IsolationGame()
    arcade.run()
