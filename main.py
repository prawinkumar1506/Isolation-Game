import pygame
import sys
import math
import copy
from enum import Enum
from typing import List, Tuple, Optional
import random

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 5
CELL_SIZE = 80
MARGIN = 80
WINDOW_WIDTH = GRID_SIZE * CELL_SIZE + 2 * MARGIN
WINDOW_HEIGHT = GRID_SIZE * CELL_SIZE + 2 * MARGIN + 150
FPS = 60

# Colors with enhanced palette
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

# Gradient colors
GRADIENT_BLUE = [(30, 60, 120), (65, 105, 225), (100, 149, 237)]
GRADIENT_RED = [(120, 30, 30), (220, 60, 60), (255, 100, 100)]
GRADIENT_GREEN = [(30, 120, 30), (60, 200, 60), (100, 255, 100)]

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

class Animation:
    def __init__(self, start_pos, end_pos, duration=500):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.duration = duration
        self.start_time = pygame.time.get_ticks()
        self.active = True
    
    def get_current_pos(self):
        current_time = pygame.time.get_ticks()
        elapsed = current_time - self.start_time
        
        if elapsed >= self.duration:
            self.active = False
            return self.end_pos
        
        progress = elapsed / self.duration
        # Smooth easing
        progress = 1 - (1 - progress) ** 3
        
        x = self.start_pos[0] + (self.end_pos[0] - self.start_pos[0]) * progress
        y = self.start_pos[1] + (self.end_pos[1] - self.start_pos[1]) * progress
        
        return (x, y)

class Particle:
    def __init__(self, x, y, color, velocity, lifetime=1000):
        self.x = x
        self.y = y
        self.color = color
        self.vx, self.vy = velocity
        self.lifetime = lifetime
        self.age = 0
        self.size = random.uniform(2, 5)
    
    def update(self, dt):
        self.x += self.vx * dt / 16.67  # 60 FPS normalization
        self.y += self.vy * dt / 16.67
        self.vy += 0.2  # Gravity
        self.age += dt
        return self.age < self.lifetime
    
    def draw(self, screen):
        alpha = max(0, 255 * (1 - self.age / self.lifetime))
        color_with_alpha = (*self.color, int(alpha))
        size = max(1, self.size * (1 - self.age / self.lifetime))
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(size))

class GameState:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.grid = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        self.player1_pos = None
        self.player2_pos = None
        self.current_player = Player.PLAYER1
        self.player1_piece = PieceType.QUEEN
        self.player2_piece = PieceType.QUEEN
        self.game_over = False
        self.winner = None
        
    def is_valid_position(self, row, col):
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size and self.grid[row][col] == 0
    
    def get_valid_moves(self, player, piece_type, position):
        if not position:
            return []
        
        row, col = position
        moves = []
        
        if piece_type == PieceType.KING:
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc
                if self.is_valid_position(new_row, new_col):
                    moves.append((new_row, new_col))
                    
        elif piece_type == PieceType.QUEEN:
            directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            for dr, dc in directions:
                for i in range(1, self.grid_size):
                    new_row, new_col = row + i*dr, col + i*dc
                    if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                        if self.grid[new_row][new_col] == 0:
                            moves.append((new_row, new_col))
                        else:
                            break
                    else:
                        break
                        
        elif piece_type == PieceType.ROOK:
            directions = [(-1,0), (1,0), (0,-1), (0,1)]
            for dr, dc in directions:
                for i in range(1, self.grid_size):
                    new_row, new_col = row + i*dr, col + i*dc
                    if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                        if self.grid[new_row][new_col] == 0:
                            moves.append((new_row, new_col))
                        else:
                            break
                    else:
                        break
                        
        elif piece_type == PieceType.KNIGHT:
            knight_moves = [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]
            for dr, dc in knight_moves:
                new_row, new_col = row + dr, col + dc
                if self.is_valid_position(new_row, new_col):
                    moves.append((new_row, new_col))
        
        elif piece_type == PieceType.BISHOP:
            directions = [(-1,-1), (-1,1), (1,-1), (1,1)]
            for dr, dc in directions:
                for i in range(1, self.grid_size):
                    new_row, new_col = row + i*dr, col + i*dc
                    if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                        if self.grid[new_row][new_col] == 0:
                            moves.append((new_row, new_col))
                        else:
                            break
                    else:
                        break
        
        elif piece_type == PieceType.PAWN:
            # Pawn moves forward one square (direction depends on player)
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
        new_state = GameState(self.grid_size)
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
    def __init__(self, depth=4):
        self.depth = depth
        self.infinity = float('inf')
    
    def evaluate(self, state):
        ai_moves_now = len(state.get_valid_moves(Player.AI, state.player2_piece, state.player2_pos))
        human_moves_now = len(state.get_valid_moves(Player.HUMAN, state.player1_piece, state.player1_pos))
        
        if ai_moves_now == 0:
            return -1000
        if human_moves_now == 0:
            return 1000
        
        ai_best_future = self.best_future_mobility(state, Player.AI)
        human_best_future = self.best_future_mobility(state, Player.HUMAN)
        
        block_bonus = self.calculate_block_bonus(state)
        
        score = (ai_moves_now - human_moves_now) + (ai_best_future - human_best_future) + block_bonus
        return score
    
    def best_future_mobility(self, state, player):
        best = 0
        piece_type = state.player2_piece if player == Player.AI else state.player1_piece
        position = state.player2_pos if player == Player.AI else state.player1_pos
        
        if not position:
            return 0
            
        valid_moves = state.get_valid_moves(player, piece_type, position)
        
        for move in valid_moves:
            new_state = state.copy()
            new_state.make_move(player, move)
            new_position = move
            mobility = len(new_state.get_valid_moves(player, piece_type, new_position))
            best = max(best, mobility)
        
        return best
    
    def calculate_block_bonus(self, state):
        if not state.player1_pos or not state.player2_pos:
            return 0
            
        human_moves = state.get_valid_moves(Player.HUMAN, state.player1_piece, state.player1_pos)
        best_human_square = None
        best_human_mobility = -1
        
        for move in human_moves:
            new_state = state.copy()
            new_state.make_move(Player.HUMAN, move)
            mobility = len(new_state.get_valid_moves(Player.HUMAN, state.player1_piece, move))
            if mobility > best_human_mobility:
                best_human_mobility = mobility
                best_human_square = move
        
        ai_moves = state.get_valid_moves(Player.AI, state.player2_piece, state.player2_pos)
        if best_human_square in ai_moves:
            return 5
        
        return 0
    
    def minimax(self, state, depth, alpha, beta, maximizing_player):
        if depth == 0 or self.is_terminal(state):
            return self.evaluate(state)
        
        if maximizing_player:
            max_eval = -self.infinity
            ai_moves = state.get_valid_moves(Player.AI, state.player2_piece, state.player2_pos)
            
            for move in ai_moves:
                new_state = state.copy()
                new_state.make_move(Player.AI, move)
                eval_score = self.minimax(new_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = self.infinity
            human_moves = state.get_valid_moves(Player.HUMAN, state.player1_piece, state.player1_pos)
            
            for move in human_moves:
                new_state = state.copy()
                new_state.make_move(Player.HUMAN, move)
                eval_score = self.minimax(new_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval
    
    def is_terminal(self, state):
        ai_moves = state.get_valid_moves(Player.AI, state.player2_piece, state.player2_pos)
        human_moves = state.get_valid_moves(Player.HUMAN, state.player1_piece, state.player1_pos)
        return len(ai_moves) == 0 or len(human_moves) == 0
    
    def get_best_move(self, state):
        best_score = -self.infinity
        best_move = None
        
        ai_moves = state.get_valid_moves(Player.AI, state.player2_piece, state.player2_pos)
        
        for move in ai_moves:
            new_state = state.copy()
            new_state.make_move(Player.AI, move)
            score = self.minimax(new_state, self.depth - 1, -self.infinity, self.infinity, False)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move

class IsolationGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Enhanced Isolation Game")
        self.clock = pygame.time.Clock()
        
        # Fonts
        self.title_font = pygame.font.Font(None, 72)
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 48)
        
        # Game state
        self.state = GameState()
        self.ai = IsolationAI(depth=4)
        self.selected_pos = None
        self.valid_moves = []
        self.game_stage = "main_menu"  # main_menu, mode_selection, piece_selection, playing, game_over
        self.game_mode = None
        self.animations = []
        self.particles = []
        
        # UI Elements
        self.hover_pos = None
        self.button_animations = {}
        self.piece_selection_phase = "player1"  # player1, player2 (for PvP)
        
        # Animation timing
        self.last_time = pygame.time.get_ticks()
        
        # Background animation
        self.bg_offset = 0
    
    def create_particles(self, x, y, color, count=10):
        for _ in range(count):
            velocity = (random.uniform(-100, 100), random.uniform(-150, -50))
            particle = Particle(x, y, color, velocity)
            self.particles.append(particle)
    
    def update_particles(self, dt):
        self.particles = [p for p in self.particles if p.update(dt)]
    
    def draw_gradient_rect(self, surface, colors, rect):
        """Draw a gradient rectangle"""
        if len(colors) < 2:
            pygame.draw.rect(surface, colors[0], rect)
            return
        
        height = rect.height
        for i in range(height):
            ratio = i / height
            # Interpolate between colors
            color_index = ratio * (len(colors) - 1)
            lower_index = int(color_index)
            upper_index = min(lower_index + 1, len(colors) - 1)
            
            if lower_index == upper_index:
                color = colors[lower_index]
            else:
                local_ratio = color_index - lower_index
                color1 = colors[lower_index]
                color2 = colors[upper_index]
                color = tuple(int(c1 + (c2 - c1) * local_ratio) for c1, c2 in zip(color1, color2))
            
            pygame.draw.line(surface, color, 
                           (rect.x, rect.y + i), 
                           (rect.x + rect.width, rect.y + i))
    
    def draw_piece_icon(self, surface, piece_type, x, y, size=30, color=BLACK):
        """Draw piece icons with enhanced 2D graphics"""
        center_x, center_y = x, y
        
        if piece_type == PieceType.KING:
            # Crown shape
            points = [
                (center_x, center_y - size//2),
                (center_x - size//3, center_y - size//4),
                (center_x - size//2, center_y),
                (center_x - size//4, center_y + size//4),
                (center_x + size//4, center_y + size//4),
                (center_x + size//2, center_y),
                (center_x + size//3, center_y - size//4)
            ]
            pygame.draw.polygon(surface, color, points)
            # Crown jewels
            pygame.draw.circle(surface, GOLD, (center_x, center_y - size//4), 3)
            
        elif piece_type == PieceType.QUEEN:
            # Crown with multiple peaks
            points = [
                (center_x, center_y - size//2),
                (center_x - size//4, center_y - size//3),
                (center_x - size//2, center_y - size//6),
                (center_x - size//3, center_y),
                (center_x, center_y - size//8),
                (center_x + size//3, center_y),
                (center_x + size//2, center_y - size//6),
                (center_x + size//4, center_y - size//3)
            ]
            pygame.draw.polygon(surface, color, points)
            # Base
            pygame.draw.rect(surface, color, (center_x - size//2, center_y, size, size//4))
            
        elif piece_type == PieceType.ROOK:
            # Castle battlements
            pygame.draw.rect(surface, color, (center_x - size//2, center_y - size//3, size, size//2))
            # Battlements
            for i in range(3):
                x_pos = center_x - size//3 + i * size//3
                pygame.draw.rect(surface, color, (x_pos, center_y - size//2, size//6, size//6))
            # Base
            pygame.draw.rect(surface, color, (center_x - size//3, center_y + size//6, 2*size//3, size//4))
            
        elif piece_type == PieceType.KNIGHT:
            # Horse head shape
            points = [
                (center_x - size//4, center_y + size//3),
                (center_x - size//2, center_y),
                (center_x - size//3, center_y - size//2),
                (center_x, center_y - size//3),
                (center_x + size//4, center_y - size//4),
                (center_x + size//3, center_y),
                (center_x + size//4, center_y + size//3)
            ]
            pygame.draw.polygon(surface, color, points)
            # Eye
            pygame.draw.circle(surface, WHITE, (center_x - size//6, center_y - size//6), 2)
            
        elif piece_type == PieceType.BISHOP:
            # Bishop mitre
            points = [
                (center_x, center_y - size//2),
                (center_x - size//4, center_y - size//6),
                (center_x - size//3, center_y + size//6),
                (center_x + size//3, center_y + size//6),
                (center_x + size//4, center_y - size//6)
            ]
            pygame.draw.polygon(surface, color, points)
            # Cross on top
            pygame.draw.line(surface, WHITE, (center_x, center_y - size//2), (center_x, center_y - size//3), 2)
            pygame.draw.line(surface, WHITE, (center_x - size//8, center_y - 3*size//8), (center_x + size//8, center_y - 3*size//8), 2)
            
        elif piece_type == PieceType.PAWN:
            # Simple pawn shape
            pygame.draw.circle(surface, color, (center_x, center_y - size//4), size//4)
            pygame.draw.rect(surface, color, (center_x - size//6, center_y - size//6, size//3, size//2))
            # Base
            pygame.draw.rect(surface, color, (center_x - size//4, center_y + size//4, size//2, size//8))
    
    def draw_animated_background(self):
        """Draw animated background"""
        self.bg_offset += 0.5
        if self.bg_offset > 50:
            self.bg_offset = 0
        
        # Gradient background
        self.draw_gradient_rect(self.screen, [(20, 30, 60), (40, 60, 120), (60, 90, 180)], 
                               pygame.Rect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT))
        
        # Moving pattern
        for i in range(0, WINDOW_WIDTH + 100, 100):
            for j in range(0, WINDOW_HEIGHT + 100, 100):
                x = i - self.bg_offset
                y = j - self.bg_offset * 0.5
                alpha = 30 + 20 * math.sin((x + y) * 0.01 + pygame.time.get_ticks() * 0.001)
                color = (int(alpha), int(alpha), int(alpha * 1.5))
                pygame.draw.circle(self.screen, color, (int(x), int(y)), 2)
    
    def draw_main_menu(self):
        """Draw enhanced main menu"""
        self.draw_animated_background()
        
        # Title with glow effect
        title_text = "ISOLATION"
        for offset in [(2, 2), (1, 1), (0, 0)]:
            color = (100, 100, 100) if offset != (0, 0) else WHITE
            text = self.title_font.render(title_text, True, color)
            rect = text.get_rect(center=(WINDOW_WIDTH//2 + offset[0], 120 + offset[1]))
            self.screen.blit(text, rect)
        
        subtitle = self.large_font.render("Strategic Board Game", True, LIGHT_GRAY)
        subtitle_rect = subtitle.get_rect(center=(WINDOW_WIDTH//2, 180))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Animated "Press any key" text
        alpha = int(127 + 127 * math.sin(pygame.time.get_ticks() * 0.003))
        color = (*WHITE[:3], alpha) if hasattr(WHITE, '__len__') and len(WHITE) > 3 else WHITE
        press_text = self.font.render("Press any key to continue", True, color)
        press_rect = press_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT - 100))
        self.screen.blit(press_text, press_rect)
        
        # Decorative elements
        for i in range(5):
            x = WINDOW_WIDTH//2 + (i - 2) * 60
            y = WINDOW_HEIGHT//2 + 50
            pulse = 1 + 0.2 * math.sin(pygame.time.get_ticks() * 0.005 + i)
            size = int(15 * pulse)
            color = [BLUE, RED, GREEN, PURPLE, ORANGE][i]
            pygame.draw.circle(self.screen, color, (x, y), size)
    
    def draw_mode_selection(self):
        """Draw game mode selection screen"""
        self.draw_animated_background()
        
        title = self.large_font.render("Select Game Mode", True, WHITE)
        title_rect = title.get_rect(center=(WINDOW_WIDTH//2, 100))
        self.screen.blit(title, title_rect)
        
        # Mode buttons
        button_width, button_height = 300, 80
        buttons = [
            {"text": "Player vs AI", "mode": GameMode.PVA, "y": 200, "color": GRADIENT_BLUE},
            {"text": "Player vs Player", "mode": GameMode.PVP, "y": 320, "color": GRADIENT_RED}
        ]
        
        mouse_pos = pygame.mouse.get_pos()
        
        for button in buttons:
            x = WINDOW_WIDTH//2 - button_width//2
            y = button["y"]
            rect = pygame.Rect(x, y, button_width, button_height)
            
            # Hover effect
            if rect.collidepoint(mouse_pos):
                self.draw_gradient_rect(self.screen, button["color"], 
                                      pygame.Rect(x-5, y-5, button_width+10, button_height+10))
                # Glow effect
                for i in range(3):
                    pygame.draw.rect(self.screen, (*button["color"][1], 50), 
                                   pygame.Rect(x-i*2, y-i*2, button_width+i*4, button_height+i*4), 2)
            else:
                self.draw_gradient_rect(self.screen, button["color"], rect)
            
            pygame.draw.rect(self.screen, WHITE, rect, 3)
            
            # Button text
            text = self.font.render(button["text"], True, WHITE)
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
            
            button["rect"] = rect
        
        # Back instruction
        back_text = self.small_font.render("Press ESC to go back", True, LIGHT_GRAY)
        self.screen.blit(back_text, (20, WINDOW_HEIGHT - 40))
        
        return buttons
    
    def draw_piece_selection(self):
        """Draw enhanced piece selection screen"""
        self.draw_animated_background()
        
        if self.game_mode == GameMode.PVP:
            if self.piece_selection_phase == "player1":
                title = "Player 1: Select Your Piece"
                color = BLUE
            else:
                title = "Player 2: Select Your Piece"
                color = RED
        else:
            title = "Select Your Piece"
            color = BLUE
        
        title_text = self.large_font.render(title, True, color)
        title_rect = title_text.get_rect(center=(WINDOW_WIDTH//2, 80))
        self.screen.blit(title_text, title_rect)
        
        # Piece selection grid
        pieces = list(PieceType)
        cols = 3
        rows = 2
        start_x = WINDOW_WIDTH//2 - (cols * 120)//2
        start_y = 150
        
        mouse_pos = pygame.mouse.get_pos()
        buttons = []
        
        for i, piece in enumerate(pieces):
            row = i // cols
            col = i % cols
            x = start_x + col * 120
            y = start_y + row * 100
            
            rect = pygame.Rect(x, y, 100, 80)
            
            # Hover effect
            if rect.collidepoint(mouse_pos):
                glow_rect = pygame.Rect(x-5, y-5, 110, 90)
                self.draw_gradient_rect(self.screen, GRADIENT_GREEN, glow_rect)
                # Particle effect on hover
                if random.random() < 0.3:
                    self.create_particles(rect.centerx, rect.centery, GREEN, 2)
            
            # Button background
            self.draw_gradient_rect(self.screen, [(80, 80, 120), (120, 120, 160)], rect)
            pygame.draw.rect(self.screen, WHITE, rect, 2)
            
            # Draw piece icon
            self.draw_piece_icon(self.screen, piece, rect.centerx, rect.centery - 15, 25, WHITE)
            
            # Piece name
            name_text = self.small_font.render(piece.value, True, WHITE)
            name_rect = name_text.get_rect(center=(rect.centerx, rect.centery + 25))
            self.screen.blit(name_text, name_rect)
            
            buttons.append({"rect": rect, "piece": piece})
        
        return buttons
    
    def draw_game_board(self):
        """Draw enhanced game board"""
        # Board shadow
        shadow_rect = pygame.Rect(MARGIN-5, MARGIN-5, 
                                 self.state.grid_size * CELL_SIZE + 10, 
                                 self.state.grid_size * CELL_SIZE + 10)
        pygame.draw.rect(self.screen, (20, 20, 20), shadow_rect)
        
        # Draw grid with enhanced visuals
        for row in range(self.state.grid_size):
            for col in range(self.state.grid_size):
                x = MARGIN + col * CELL_SIZE
                y = MARGIN + row * CELL_SIZE
                rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
                
                # Determine cell appearance
                if self.state.grid[row][col] == -1:  # Blocked
                    self.draw_gradient_rect(self.screen, [(40, 40, 40), (20, 20, 20)], rect)
                    # X pattern for blocked cells
                    pygame.draw.line(self.screen, RED, (x+10, y+10), (x+CELL_SIZE-10, y+CELL_SIZE-10), 3)
                    pygame.draw.line(self.screen, RED, (x+CELL_SIZE-10, y+10), (x+10, y+CELL_SIZE-10), 3)
                elif (row, col) == self.selected_pos:
                    # Pulsing selection effect
                    pulse = 1 + 0.3 * math.sin(pygame.time.get_ticks() * 0.008)
                    glow_size = int(5 * pulse)
                    glow_rect = pygame.Rect(x-glow_size, y-glow_size, 
                                          CELL_SIZE+2*glow_size, CELL_SIZE+2*glow_size)
                    self.draw_gradient_rect(self.screen, GRADIENT_GREEN, glow_rect)
                    self.draw_gradient_rect(self.screen, [(255, 255, 200), (255, 255, 150)], rect)
                elif (row, col) in self.valid_moves:
                    # Animated valid move indicators
                    pulse = 0.7 + 0.3 * math.sin(pygame.time.get_ticks() * 0.01 + row + col)
                    color_intensity = int(100 + 100 * pulse)
                    self.draw_gradient_rect(self.screen, 
                                          [(color_intensity, 255, color_intensity), 
                                           (color_intensity//2, 200, color_intensity//2)], rect)
                else:
                    # Checkerboard pattern
                    if (row + col) % 2 == 0:
                        self.draw_gradient_rect(self.screen, [(240, 240, 240), (220, 220, 220)], rect)
                    else:
                        self.draw_gradient_rect(self.screen, [(200, 200, 200), (180, 180, 180)], rect)
                
                # Grid lines
                pygame.draw.rect(self.screen, BLACK, rect, 2)
                
                # Coordinate labels
                if row == self.state.grid_size - 1:  # Bottom row
                    coord_text = self.small_font.render(str(col), True, DARK_GRAY)
                    self.screen.blit(coord_text, (x + CELL_SIZE - 15, y + CELL_SIZE + 5))
                if col == 0:  # Left column
                    coord_text = self.small_font.render(str(row), True, DARK_GRAY)
                    self.screen.blit(coord_text, (x - 20, y + 5))
        
        # Draw pieces with enhanced graphics
        self.draw_pieces()
        
        # Draw particles
        for particle in self.particles:
            particle.draw(self.screen)
    
    def draw_pieces(self):
        """Draw pieces with enhanced 2D graphics and animations"""
        if self.state.player1_pos:
            row, col = self.state.player1_pos
            x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
            y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2
            
            # Piece shadow
            shadow_offset = 3
            pygame.draw.circle(self.screen, (0, 0, 0, 100), 
                             (x + shadow_offset, y + shadow_offset), CELL_SIZE // 2 - 5)
            
            # Main piece circle with gradient
            colors = GRADIENT_BLUE if self.game_mode == GameMode.PVA else GRADIENT_BLUE
            for i in range(3):
                radius = CELL_SIZE // 2 - 5 - i * 5
                color_idx = min(i, len(colors) - 1)
                if radius > 0:
                    pygame.draw.circle(self.screen, colors[color_idx], (x, y), radius)
            
            # Outer ring
            pygame.draw.circle(self.screen, WHITE, (x, y), CELL_SIZE // 2 - 5, 3)
            
            # Piece icon
            self.draw_piece_icon(self.screen, self.state.player1_piece, x, y, 20, WHITE)
            
            # Player indicator
            indicator_text = "P1" if self.game_mode == GameMode.PVP else "YOU"
            text = self.small_font.render(indicator_text, True, WHITE)
            text_rect = text.get_rect(center=(x, y + 30))
            self.screen.blit(text, text_rect)
        
        if self.state.player2_pos:
            row, col = self.state.player2_pos
            x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
            y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2
            
            # Piece shadow
            shadow_offset = 3
            pygame.draw.circle(self.screen, (0, 0, 0, 100), 
                             (x + shadow_offset, y + shadow_offset), CELL_SIZE // 2 - 5)
            
            # Main piece circle with gradient
            colors = GRADIENT_RED
            for i in range(3):
                radius = CELL_SIZE // 2 - 5 - i * 5
                color_idx = min(i, len(colors) - 1)
                if radius > 0:
                    pygame.draw.circle(self.screen, colors[color_idx], (x, y), radius)
            
            # Outer ring
            pygame.draw.circle(self.screen, WHITE, (x, y), CELL_SIZE // 2 - 5, 3)
            
            # Piece icon
            self.draw_piece_icon(self.screen, self.state.player2_piece, x, y, 20, WHITE)
            
            # Player indicator
            indicator_text = "P2" if self.game_mode == GameMode.PVP else "AI"
            text = self.small_font.render(indicator_text, True, WHITE)
            text_rect = text.get_rect(center=(x, y + 30))
            self.screen.blit(text, text_rect)
    
    def draw_game_ui(self):
        """Draw enhanced game UI"""
        # Top panel with gradient background
        panel_rect = pygame.Rect(0, 0, WINDOW_WIDTH, MARGIN - 10)
        self.draw_gradient_rect(self.screen, [(60, 80, 120), (40, 60, 100)], panel_rect)
        
        # Current player indicator with glow effect
        if self.state.current_player == Player.PLAYER1:
            player_text = "Your Turn" if self.game_mode == GameMode.PVA else "Player 1's Turn"
            color = BLUE
        else:
            player_text = "AI Turn" if self.game_mode == GameMode.PVA else "Player 2's Turn"
            color = RED
        
        # Glowing text effect
        for offset in [(2, 2), (1, 1), (0, 0)]:
            text_color = DARK_GRAY if offset != (0, 0) else WHITE
            text = self.font.render(player_text, True, text_color)
            self.screen.blit(text, (20 + offset[0], 20 + offset[1]))
        
        # Move counter and game info
        move_info = f"Moves: P1({len(self.state.get_valid_moves(Player.PLAYER1, self.state.player1_piece, self.state.player1_pos))}) "
        move_info += f"P2({len(self.state.get_valid_moves(Player.PLAYER2, self.state.player2_piece, self.state.player2_pos))})"
        
        info_text = self.small_font.render(move_info, True, WHITE)
        self.screen.blit(info_text, (WINDOW_WIDTH - 250, 25))
        
        # Piece information
        piece_info = f"Pieces: {self.state.player1_piece.value} vs {self.state.player2_piece.value}"
        piece_text = self.small_font.render(piece_info, True, LIGHT_GRAY)
        self.screen.blit(piece_text, (20, 50))
        
        # Bottom panel
        bottom_panel = pygame.Rect(0, WINDOW_HEIGHT - 60, WINDOW_WIDTH, 60)
        self.draw_gradient_rect(self.screen, [(40, 60, 100), (60, 80, 120)], bottom_panel)
        
        # Instructions
        if self.selected_pos:
            instruction = "Click a highlighted square to move"
            color = GREEN
        else:
            instruction = "Click your piece to select it"
            color = WHITE
        
        inst_text = self.small_font.render(instruction, True, color)
        self.screen.blit(inst_text, (20, WINDOW_HEIGHT - 40))
        
        # Controls
        controls = "ESC: Menu | R: Restart"
        control_text = self.small_font.render(controls, True, LIGHT_GRAY)
        control_rect = control_text.get_rect(right=WINDOW_WIDTH - 20, y=WINDOW_HEIGHT - 40)
        self.screen.blit(control_text, control_rect)
    
    def draw_game_over(self):
        """Draw enhanced game over screen"""
        # Semi-transparent overlay
        overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 0))
        self.screen.blit(overlay, (0, 0))
        
        # Winner announcement with effects
        if self.state.winner == Player.PLAYER1:
            winner_text = "You Win!" if self.game_mode == GameMode.PVA else "Player 1 Wins!"
            color = GRADIENT_BLUE
        else:
            winner_text = "AI Wins!" if self.game_mode == GameMode.PVA else "Player 2 Wins!"
            color = GRADIENT_RED
        
        # Animated winner text
        scale = 1 + 0.1 * math.sin(pygame.time.get_ticks() * 0.005)
        winner_font = pygame.font.Font(None, int(72 * scale))
        
        # Text with glow
        for offset in [(4, 4), (2, 2), (0, 0)]:
            text_color = color[0] if offset != (0, 0) else WHITE
            text = winner_font.render(winner_text, True, text_color)
            rect = text.get_rect(center=(WINDOW_WIDTH//2 + offset[0], WINDOW_HEIGHT//2 - 50 + offset[1]))
            self.screen.blit(text, rect)
        
        # Celebration particles
        if random.random() < 0.5:
            center_x, center_y = WINDOW_WIDTH//2, WINDOW_HEIGHT//2 - 50
            self.create_particles(center_x + random.randint(-100, 100), 
                                center_y + random.randint(-50, 50), 
                                color[1], 3)
        
        # Game stats
        stats = [
            f"Final Moves: Player 1 had {len(self.state.get_valid_moves(Player.PLAYER1, self.state.player1_piece, self.state.player1_pos))} moves",
            f"Final Moves: Player 2 had {len(self.state.get_valid_moves(Player.PLAYER2, self.state.player2_piece, self.state.player2_pos))} moves",
            f"Pieces Used: {self.state.player1_piece.value} vs {self.state.player2_piece.value}"
        ]
        
        for i, stat in enumerate(stats):
            stat_text = self.small_font.render(stat, True, WHITE)
            stat_rect = stat_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 20 + i * 25))
            self.screen.blit(stat_text, stat_rect)
        
        # Restart options
        restart_text = self.font.render("Press R to Restart or ESC for Menu", True, YELLOW)
        restart_rect = restart_text.get_rect(center=(WINDOW_WIDTH//2, WINDOW_HEIGHT//2 + 120))
        self.screen.blit(restart_text, restart_rect)
    
    def setup_game(self):
        """Initialize game positions"""
        # Place player 1 at bottom-left
        self.state.player1_pos = (self.state.grid_size - 1, 0)
        self.state.grid[self.state.grid_size - 1][0] = 1
        
        # Place player 2 at top-right
        self.state.player2_pos = (0, self.state.grid_size - 1)
        self.state.grid[0][self.state.grid_size - 1] = 2
        
        # Create entrance particles
        self.create_particles(MARGIN + CELL_SIZE//2, 
                            MARGIN + (self.state.grid_size - 1) * CELL_SIZE + CELL_SIZE//2, 
                            BLUE, 15)
        self.create_particles(MARGIN + (self.state.grid_size - 1) * CELL_SIZE + CELL_SIZE//2, 
                            MARGIN + CELL_SIZE//2, 
                            RED, 15)
        
        self.game_stage = "playing"
    
    def handle_click(self, pos):
        """Handle mouse clicks with enhanced feedback"""
        x, y = pos
        
        if self.game_stage == "mode_selection":
            buttons = self.draw_mode_selection()  # Get buttons from draw function
            for button in buttons:
                if button["rect"].collidepoint(x, y):
                    self.game_mode = button["mode"]
                    self.game_stage = "piece_selection"
                    # Click particle effect
                    self.create_particles(x, y, WHITE, 8)
                    return
        
        elif self.game_stage == "piece_selection":
            buttons = self.draw_piece_selection()  # Get buttons from draw function
            for button in buttons:
                if button["rect"].collidepoint(x, y):
                    if self.game_mode == GameMode.PVP and self.piece_selection_phase == "player1":
                        self.state.player1_piece = button["piece"]
                        self.piece_selection_phase = "player2"
                        # Selection particle effect
                        self.create_particles(button["rect"].centerx, button["rect"].centery, BLUE, 10)
                    elif self.game_mode == GameMode.PVP and self.piece_selection_phase == "player2":
                        self.state.player2_piece = button["piece"]
                        self.setup_game()
                        # Selection particle effect
                        self.create_particles(button["rect"].centerx, button["rect"].centery, RED, 10)
                    else:  # PvA mode
                        self.state.player1_piece = button["piece"]
                        self.state.player2_piece = button["piece"]  # AI uses same piece
                        self.setup_game()
                        # Selection particle effect
                        self.create_particles(button["rect"].centerx, button["rect"].centery, GREEN, 10)
                    return
        
        elif self.game_stage == "playing":
            col = (x - MARGIN) // CELL_SIZE
            row = (y - MARGIN) // CELL_SIZE
            
            if 0 <= row < self.state.grid_size and 0 <= col < self.state.grid_size:
                if self.state.current_player == Player.PLAYER1:
                    if (row, col) == self.state.player1_pos:
                        # Select piece
                        self.selected_pos = (row, col)
                        self.valid_moves = self.state.get_valid_moves(Player.PLAYER1, 
                                                                    self.state.player1_piece, 
                                                                    self.state.player1_pos)
                        # Selection particle effect
                        self.create_particles(x, y, BLUE, 5)
                    elif (row, col) in self.valid_moves:
                        # Make move with animation
                        old_pos = self.state.player1_pos
                        self.state.make_move(Player.PLAYER1, (row, col))
                        
                        # Move particle effect
                        self.create_particles(x, y, GREEN, 12)
                        
                        self.selected_pos = None
                        self.valid_moves = []
                        
                        if self.check_game_over():
                            return
                        
                        self.state.current_player = Player.PLAYER2
                    else:
                        # Invalid selection
                        self.selected_pos = None
                        self.valid_moves = []
                
                elif self.state.current_player == Player.PLAYER2 and self.game_mode == GameMode.PVP:
                    if (row, col) == self.state.player2_pos:
                        # Select piece
                        self.selected_pos = (row, col)
                        self.valid_moves = self.state.get_valid_moves(Player.PLAYER2, 
                                                                    self.state.player2_piece, 
                                                                    self.state.player2_pos)
                        # Selection particle effect
                        self.create_particles(x, y, RED, 5)
                    elif (row, col) in self.valid_moves:
                        # Make move
                        self.state.make_move(Player.PLAYER2, (row, col))
                        
                        # Move particle effect
                        self.create_particles(x, y, GREEN, 12)
                        
                        self.selected_pos = None
                        self.valid_moves = []
                        
                        if self.check_game_over():
                            return
                        
                        self.state.current_player = Player.PLAYER1
                    else:
                        # Invalid selection
                        self.selected_pos = None
                        self.valid_moves = []
    
    def ai_move(self):
        """Execute AI move with visual effects"""
        if self.state.current_player == Player.PLAYER2 and self.game_mode == GameMode.PVA:
            # Convert to AI-compatible state
            ai_state = self.state.copy()
            ai_state.current_player = Player.AI
            ai_state.human_pos = self.state.player1_pos
            ai_state.ai_pos = self.state.player2_pos
            ai_state.human_piece = self.state.player1_piece
            ai_state.ai_piece = self.state.player2_piece
            
            best_move = self.ai.get_best_move(ai_state)
            
            if best_move:
                # AI move particle effect
                row, col = best_move
                x = MARGIN + col * CELL_SIZE + CELL_SIZE // 2
                y = MARGIN + row * CELL_SIZE + CELL_SIZE // 2
                self.create_particles(x, y, RED, 15)
                
                self.state.make_move(Player.PLAYER2, best_move)
                
                if self.check_game_over():
                    return
                
                self.state.current_player = Player.PLAYER1
    
    def check_game_over(self):
        """Check if game is over"""
        player1_moves = self.state.get_valid_moves(Player.PLAYER1, 
                                                  self.state.player1_piece, 
                                                  self.state.player1_pos)
        player2_moves = self.state.get_valid_moves(Player.PLAYER2, 
                                                  self.state.player2_piece, 
                                                  self.state.player2_pos)
        
        if len(player1_moves) == 0:
            self.state.game_over = True
            self.state.winner = Player.PLAYER2
            self.game_stage = "game_over"
            # Victory particle explosion
            for _ in range(30):
                x = random.randint(100, WINDOW_WIDTH - 100)
                y = random.randint(100, WINDOW_HEIGHT - 100)
                self.create_particles(x, y, RED, 3)
            return True
        elif len(player2_moves) == 0:
            self.state.game_over = True
            self.state.winner = Player.PLAYER1
            self.game_stage = "game_over"
            # Victory particle explosion
            for _ in range(30):
                x = random.randint(100, WINDOW_WIDTH - 100)
                y = random.randint(100, WINDOW_HEIGHT - 100)
                self.create_particles(x, y, BLUE, 3)
            return True
        
        return False
    
    def reset_game(self):
        """Reset game to initial state"""
        self.state = GameState()
        self.selected_pos = None
        self.valid_moves = []
        self.game_stage = "mode_selection"
        self.piece_selection_phase = "player1"
        self.particles.clear()
        self.animations.clear()
    
    def run(self):
        """Main game loop with enhanced timing"""
        running = True
        
        while running:
            current_time = pygame.time.get_ticks()
            dt = current_time - self.last_time
            self.last_time = current_time
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.game_stage in ["mode_selection", "piece_selection"]:
                            if self.game_stage == "piece_selection" and self.piece_selection_phase == "player2":
                                self.piece_selection_phase = "player1"
                            else:
                                self.game_stage = "main_menu"
                        else:
                            self.reset_game()
                    
                    elif event.key == pygame.K_r and self.game_stage == "game_over":
                        self.reset_game()
                    
                    elif self.game_stage == "main_menu":
                        self.game_stage = "mode_selection"
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
            
            # Update particles and animations
            self.update_particles(dt)
            
            # AI move logic
            if (self.game_stage == "playing" and 
                self.state.current_player == Player.PLAYER2 and 
                self.game_mode == GameMode.PVA):
                self.ai_move()
            
            # Render everything
            self.screen.fill(BLACK)
            
            if self.game_stage == "main_menu":
                self.draw_main_menu()
            elif self.game_stage == "mode_selection":
                self.draw_mode_selection()
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
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = IsolationGame()
    game.run()