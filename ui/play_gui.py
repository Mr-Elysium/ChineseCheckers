#!/usr/bin/env python3
"""
Pygame GUI Interface for Chinese Checkers
Beautiful hexagonal board visualization with mouse controls
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import math
import torch
import numpy as np
import cc_core
from network import ChineseCheckersNet
from utils import hex_to_tensor, ROWS, COLS

try:
    import pygame
    from pygame import gfxdraw
except ImportError:
    print("Error: pygame not installed. Install with: pip install pygame")
    sys.exit(1)


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
DARK_GRAY = (100, 100, 100)
PLAYER1_COLOR = (50, 120, 220)  # Blue
PLAYER2_COLOR = (220, 50, 50)   # Red
HIGHLIGHT_COLOR = (255, 215, 0)  # Gold
LEGAL_MOVE_COLOR = (100, 200, 100)  # Green
SELECTED_COLOR = (255, 165, 0)  # Orange
BOARD_BG = (245, 245, 220)  # Beige


class HexagonalBoard:
    """Handles hexagonal board rendering and interaction."""
    
    def __init__(self, screen, offset_x=450, offset_y=350, hex_size=26):
        self.screen = screen
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.hex_size = hex_size
        self.hexagons = []  # List of (center_x, center_y, idx)
        self.selected_idx = None
        self.legal_move_indices = []
        self.hovered_idx = None
        
        self._build_hexagons()
    
    def _build_hexagons(self):
        """Build hexagon positions for all 121 board positions."""
        self.hexagons = []
        
        for idx in range(121):
            row, col = ROWS[idx], COLS[idx]
            x, y = self._grid_to_pixel(row, col)
            self.hexagons.append((x, y, idx))
    
    def _grid_to_pixel(self, row, col):
        """Convert grid coordinates using axial/cube hex geometry."""
        # For pointy-top hexagons touching at edges:
        # Distance between centers when touching = 2 * hex_size
        # Use axial coordinates with proper angular positioning
        
        sqrt3 = math.sqrt(3)
        
        # Convert grid row/col to axial hex coordinates
        # Center at (8, 8) in the 17x17 grid
        q = col - 8
        r = row - 8
        
        # Pointy-top hex layout:
        # x = hex_size * sqrt(3) * (q + r/2)
        # y = hex_size * 3/2 * r
        
        x = self.offset_x + self.hex_size * sqrt3 * (q + r / 2.0)
        y = self.offset_y + self.hex_size * 1.5 * r
        
        return int(x), int(y)
    
    def _draw_hexagon(self, x, y, color, filled=True, border_color=None, border_width=2):
        """Draw a pointy-top hexagon at pixel position (x, y)."""
        points = []
        # Pointy-top hexagon: points at top/bottom
        # Vertices at 30°, 90°, 150°, 210°, 270°, 330° from center
        # Start at top-right and go clockwise
        for i in range(6):
            angle = math.pi / 6 + math.pi / 3 * i  # 30°, 90°, 150°, 210°, 270°, 330°
            px = x + self.hex_size * math.cos(angle)
            py = y + self.hex_size * math.sin(angle)
            points.append((px, py))
        
        if filled:
            pygame.draw.polygon(self.screen, color, points)
        
        if border_color:
            pygame.draw.polygon(self.screen, border_color, points, border_width)
    
    def draw(self, board_state):
        """Draw the entire board."""
        grid = board_state.get_grid()
        
        for x, y, idx in self.hexagons:
            piece = grid[idx]
            
            # Determine colors
            if idx == self.selected_idx:
                fill_color = SELECTED_COLOR
                border_color = BLACK
                border_width = 3
            elif idx == self.hovered_idx:
                fill_color = HIGHLIGHT_COLOR
                border_color = BLACK
                border_width = 2
            elif idx in self.legal_move_indices:
                fill_color = LEGAL_MOVE_COLOR
                border_color = DARK_GRAY
                border_width = 2
            elif piece == 0:
                fill_color = WHITE
                border_color = GRAY
                border_width = 1
            elif piece == 1:
                fill_color = PLAYER1_COLOR
                border_color = BLACK
                border_width = 2
            elif piece == 2:
                fill_color = PLAYER2_COLOR
                border_color = BLACK
                border_width = 2
            else:
                fill_color = LIGHT_GRAY
                border_color = GRAY
                border_width = 1
            
            self._draw_hexagon(x, y, fill_color, filled=True, 
                             border_color=border_color, border_width=border_width)
            
            # Draw legal move indicator (small circle)
            if idx in self.legal_move_indices and piece == 0:
                pygame.draw.circle(self.screen, DARK_GRAY, (x, y), 5)
    
    def get_hex_at_position(self, mouse_x, mouse_y):
        """Find which hexagon was clicked."""
        for x, y, idx in self.hexagons:
            dist = math.sqrt((mouse_x - x)**2 + (mouse_y - y)**2)
            if dist < self.hex_size:
                return idx
        return None
    
    def set_selected(self, idx):
        """Set the selected hexagon."""
        self.selected_idx = idx
    
    def set_legal_moves(self, move_indices):
        """Set legal move destination indices."""
        self.legal_move_indices = move_indices
    
    def set_hovered(self, idx):
        """Set the hovered hexagon."""
        self.hovered_idx = idx


class GameGUI:
    """Main game GUI controller."""
    
    def __init__(self, model_path=None, mcts_iterations=200, device='cpu', game_mode='human_vs_ai_p1'):
        pygame.init()
        
        self.width = 1300
        self.height = 850
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Chinese Checkers - AlphaZero")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 48)
        
        # Game state
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mcts_iterations = mcts_iterations
        self.game_mode = game_mode  # 'human_vs_ai_p1', 'human_vs_ai_p2', 'human_vs_human'
        
        cc_core.MoveGen.initialize()
        self.board = cc_core.Board(2)
        self.mcts = cc_core.MCTS(1.41)
        
        self.hex_board = HexagonalBoard(self.screen, offset_x=450, offset_y=350, hex_size=26)
        
        # Determine human players based on mode
        if game_mode == 'human_vs_ai_p1':
            self.human_players = [1]  # Human is Player 1
        elif game_mode == 'human_vs_ai_p2':
            self.human_players = [2]  # Human is Player 2
        else:  # human_vs_human
            self.human_players = [1, 2]  # Both are human
        
        self.current_state = "playing"  # playing, game_over, menu
        self.move_count = 0
        self.max_moves = 400
        self.winner = None
        self.show_menu = True
        
        # Load model
        self.model = None
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            config = checkpoint.get('config', {})
            
            self.model = ChineseCheckersNet(
                num_players=config.get('num_players', 2),
                num_res_blocks=config.get('num_res_blocks', 10),
                num_channels=config.get('num_channels', 128)
            ).to(self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✓ Loaded model: {config.get('num_res_blocks', 10)} blocks, {config.get('num_channels', 128)} channels")
        else:
            print("⚠ No model loaded - AI will use random policy")
    
    def handle_click(self, pos):
        """Handle mouse click."""
        # Handle menu clicks
        if self.show_menu:
            self.handle_menu_click(pos)
            return
        
        if self.current_state != "playing":
            return
        
        # Check if current player is human
        if self.board.get_current_player() not in self.human_players:
            return
        
        clicked_idx = self.hex_board.get_hex_at_position(pos[0], pos[1])
        
        if clicked_idx is None:
            # Clicked outside board, deselect
            self.hex_board.set_selected(None)
            self.hex_board.set_legal_moves([])
            return
        
        grid = self.board.get_grid()
        current_player = self.board.get_current_player()
        
        # If nothing selected, select piece
        if self.hex_board.selected_idx is None:
            if grid[clicked_idx] == current_player:
                self.hex_board.set_selected(clicked_idx)
                
                # Get legal moves from this position
                legal_moves = cc_core.MoveGen.get_legal_moves(self.board)
                legal_destinations = [m.to_idx for m in legal_moves if m.from_idx == clicked_idx]
                self.hex_board.set_legal_moves(legal_destinations)
        else:
            # Something is selected
            from_idx = self.hex_board.selected_idx
            
            # Check if clicked on legal destination
            if clicked_idx in self.hex_board.legal_move_indices:
                # Execute move
                move = cc_core.Move(from_idx, clicked_idx)
                self.board.apply_move(move)
                self.move_count += 1
                
                # Clear selection
                self.hex_board.set_selected(None)
                self.hex_board.set_legal_moves([])
                
                # Check if game over
                if self.board.is_terminal():
                    self.current_state = "game_over"
                    rewards = self.board.get_rewards()
                    self.winner = 1 if rewards[0] > rewards[1] else 2
                elif self.move_count >= self.max_moves:
                    self.current_state = "game_over"
                    self.winner = None
            else:
                # Clicked on own piece, switch selection
                if grid[clicked_idx] == current_player:
                    self.hex_board.set_selected(clicked_idx)
                    legal_moves = cc_core.MoveGen.get_legal_moves(self.board)
                    legal_destinations = [m.to_idx for m in legal_moves if m.from_idx == clicked_idx]
                    self.hex_board.set_legal_moves(legal_destinations)
                else:
                    # Clicked elsewhere, deselect
                    self.hex_board.set_selected(None)
                    self.hex_board.set_legal_moves([])
    
    def handle_menu_click(self, pos):
        """Handle clicks on the menu."""
        # Menu buttons
        button_width = 400
        button_height = 60
        start_y = 250
        spacing = 80
        
        center_x = self.width // 2
        
        buttons = [
            ('human_vs_ai_p1', "Play as Player 1 (Blue) vs AI"),
            ('human_vs_ai_p2', "Play as Player 2 (Red) vs AI"),
            ('human_vs_human', "Human vs Human")
        ]
        
        for i, (mode, _) in enumerate(buttons):
            button_rect = pygame.Rect(
                center_x - button_width // 2,
                start_y + i * spacing,
                button_width,
                button_height
            )
            
            if button_rect.collidepoint(pos):
                self.start_game(mode)
                break
    
    def start_game(self, mode):
        """Start a new game with the selected mode."""
        self.game_mode = mode
        self.show_menu = False
        
        # Reset game state
        self.board = cc_core.Board(2)
        self.mcts = cc_core.MCTS(1.41)
        self.move_count = 0
        self.winner = None
        self.current_state = "playing"
        self.hex_board.set_selected(None)
        self.hex_board.set_legal_moves([])
        
        # Set human players
        if mode == 'human_vs_ai_p1':
            self.human_players = [1]
        elif mode == 'human_vs_ai_p2':
            self.human_players = [2]
        else:
            self.human_players = [1, 2]
    
    def handle_hover(self, pos):
        """Handle mouse hover."""
        if not self.show_menu:
            hovered_idx = self.hex_board.get_hex_at_position(pos[0], pos[1])
            self.hex_board.set_hovered(hovered_idx)
    
    def ai_move(self):
        """Execute AI move."""
        def predictor(board_array):
            if self.model is None:
                policy = [1.0 / 121.0] * 121
                value = [0.0, 0.0]
                return policy, value
            
            current_player = self.board.get_current_player()
            state_tensor = hex_to_tensor(np.array(board_array), current_player)
            
            with torch.no_grad():
                state_tensor = state_tensor.unsqueeze(0).to(self.device)
                policy_logits, value = self.model(state_tensor)
                
                policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
                value_vec = value[0].cpu().numpy()
            
            if len(value_vec) == 1:
                value_output = [value_vec[0], -value_vec[0]]
            else:
                value_output = value_vec.tolist()
            
            return policy.tolist(), value_output
        
        # Run MCTS
        self.mcts.search(self.mcts_iterations, self.board, predictor)
        best_move = self.mcts.get_best_move()
        
        self.board.apply_move(best_move)
        self.move_count += 1
        
        # Check if game over
        if self.board.is_terminal():
            self.current_state = "game_over"
            rewards = self.board.get_rewards()
            self.winner = 1 if rewards[0] > rewards[1] else 2
        elif self.move_count >= self.max_moves:
            self.current_state = "game_over"
            self.winner = None
        else:
            self.current_state = "playing"
    
    def draw_menu(self):
        """Draw the main menu."""
        # Background
        self.screen.fill(BOARD_BG)
        
        # Title
        title = self.title_font.render("Chinese Checkers", True, BLACK)
        title_rect = title.get_rect(center=(self.width // 2, 100))
        self.screen.blit(title, title_rect)
        
        subtitle = self.small_font.render("AlphaZero AI", True, DARK_GRAY)
        subtitle_rect = subtitle.get_rect(center=(self.width // 2, 150))
        self.screen.blit(subtitle, subtitle_rect)
        
        # Menu buttons
        button_width = 400
        button_height = 60
        start_y = 250
        spacing = 80
        
        center_x = self.width // 2
        mouse_pos = pygame.mouse.get_pos()
        
        buttons = [
            ('human_vs_ai_p1', "Play as Player 1 (Blue) vs AI", PLAYER1_COLOR),
            ('human_vs_ai_p2', "Play as Player 2 (Red) vs AI", PLAYER2_COLOR),
            ('human_vs_human', "Human vs Human", (100, 100, 200))
        ]
        
        for i, (mode, text, color) in enumerate(buttons):
            button_rect = pygame.Rect(
                center_x - button_width // 2,
                start_y + i * spacing,
                button_width,
                button_height
            )
            
            # Check hover
            is_hovered = button_rect.collidepoint(mouse_pos)
            button_color = color if is_hovered else LIGHT_GRAY
            text_color = WHITE if is_hovered else BLACK
            
            # Draw button
            pygame.draw.rect(self.screen, button_color, button_rect, border_radius=10)
            pygame.draw.rect(self.screen, BLACK, button_rect, 3, border_radius=10)
            
            # Draw text
            button_text = self.small_font.render(text, True, text_color)
            text_rect = button_text.get_rect(center=button_rect.center)
            self.screen.blit(button_text, text_rect)
    
    def draw_ui(self):
        """Draw UI elements (info panel, buttons, etc.)."""
        # Info panel background
        panel_x = 1000
        pygame.draw.rect(self.screen, LIGHT_GRAY, (panel_x, 0, 300, self.height))
        
        y_offset = 30
        
        # Title
        title = self.font.render("Chinese Checkers", True, BLACK)
        self.screen.blit(title, (panel_x + 20, y_offset))
        y_offset += 50
        
        # Game mode
        mode_text = ""
        if self.game_mode == 'human_vs_ai_p1':
            mode_text = "You (Blue) vs AI (Red)"
        elif self.game_mode == 'human_vs_ai_p2':
            mode_text = "AI (Blue) vs You (Red)"
        else:
            mode_text = "Human vs Human"
        
        text = self.small_font.render(mode_text, True, DARK_GRAY)
        self.screen.blit(text, (panel_x + 20, y_offset))
        y_offset += 40
        
        # Separator
        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x + 10, y_offset, 280, 2))
        y_offset += 20
        
        # Current player
        current_player = self.board.get_current_player()
        player_text = f"Current: Player {current_player}"
        player_color = PLAYER1_COLOR if current_player == 1 else PLAYER2_COLOR
        text = self.small_font.render(player_text, True, player_color)
        self.screen.blit(text, (panel_x + 20, y_offset))
        y_offset += 35
        
        # Move count
        move_text = f"Move: {self.move_count}/{self.max_moves}"
        text = self.small_font.render(move_text, True, BLACK)
        self.screen.blit(text, (panel_x + 20, y_offset))
        y_offset += 35
        
        # State
        if self.current_state == "playing":
            if current_player in self.human_players:
                state_text = "Your turn!"
                state_color = (0, 150, 0)
            else:
                state_text = "AI thinking..."
                state_color = (200, 100, 0)
        else:
            state_text = "Game Over"
            state_color = DARK_GRAY
        
        text = self.small_font.render(state_text, True, state_color)
        self.screen.blit(text, (panel_x + 20, y_offset))
        y_offset += 50
        
        # Legend
        pygame.draw.rect(self.screen, DARK_GRAY, (panel_x + 10, y_offset, 280, 2))
        y_offset += 20
        
        legend_title = self.small_font.render("Legend:", True, BLACK)
        self.screen.blit(legend_title, (panel_x + 20, y_offset))
        y_offset += 30
        
        # Player 1
        pygame.draw.circle(self.screen, PLAYER1_COLOR, (panel_x + 30, y_offset + 10), 10)
        p1_label = "Player 1 (You)" if 1 in self.human_players else "Player 1 (AI)"
        text = self.small_font.render(p1_label, True, BLACK)
        self.screen.blit(text, (panel_x + 50, y_offset))
        y_offset += 30
        
        # Player 2
        pygame.draw.circle(self.screen, PLAYER2_COLOR, (panel_x + 30, y_offset + 10), 10)
        p2_label = "Player 2 (You)" if 2 in self.human_players else "Player 2 (AI)"
        text = self.small_font.render(p2_label, True, BLACK)
        self.screen.blit(text, (panel_x + 50, y_offset))
        y_offset += 30
        
        # Selected
        pygame.draw.circle(self.screen, SELECTED_COLOR, (panel_x + 30, y_offset + 10), 10)
        text = self.small_font.render("Selected", True, BLACK)
        self.screen.blit(text, (panel_x + 50, y_offset))
        y_offset += 30
        
        # Legal move
        pygame.draw.circle(self.screen, LEGAL_MOVE_COLOR, (panel_x + 30, y_offset + 10), 10)
        text = self.small_font.render("Legal Move", True, BLACK)
        self.screen.blit(text, (panel_x + 50, y_offset))
        y_offset += 50
        
        # New Game button
        button_rect = pygame.Rect(panel_x + 20, y_offset, 260, 40)
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = button_rect.collidepoint(mouse_pos)
        button_color = (100, 150, 200) if is_hovered else (150, 150, 150)
        
        pygame.draw.rect(self.screen, button_color, button_rect, border_radius=5)
        pygame.draw.rect(self.screen, BLACK, button_rect, 2, border_radius=5)
        
        button_text = self.small_font.render("New Game", True, WHITE if is_hovered else BLACK)
        text_rect = button_text.get_rect(center=button_rect.center)
        self.screen.blit(button_text, text_rect)
        
        # Store button rect for click detection
        self.new_game_button = button_rect
        y_offset += 60
        
        # Game over message
        if self.current_state == "game_over":
            pygame.draw.rect(self.screen, DARK_GRAY, (panel_x + 10, y_offset, 280, 2))
            y_offset += 20
            
            if self.winner in self.human_players and len(self.human_players) == 1:
                msg = "You Win!"
                color = PLAYER1_COLOR if self.winner == 1 else PLAYER2_COLOR
            elif self.winner is None:
                msg = "Draw"
                color = DARK_GRAY
            elif len(self.human_players) == 2:
                msg = f"Player {self.winner} Wins!"
                color = PLAYER1_COLOR if self.winner == 1 else PLAYER2_COLOR
            else:
                msg = "AI Wins"
                color = PLAYER2_COLOR if self.winner == 2 else PLAYER1_COLOR
            
            text = self.font.render(msg, True, color)
            text_rect = text.get_rect(center=(panel_x + 150, y_offset + 30))
            self.screen.blit(text, text_rect)
    
    def run(self):
        """Main game loop."""
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Check new game button
                    if not self.show_menu and hasattr(self, 'new_game_button') and self.new_game_button.collidepoint(event.pos):
                        self.show_menu = True
                    else:
                        self.handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_hover(event.pos)
            
            # AI move (only if not human vs human and current player is AI)
            if not self.show_menu and self.current_state == "playing":
                current_player = self.board.get_current_player()
                if current_player not in self.human_players:
                    self.ai_move()
            
            # Draw everything
            if self.show_menu:
                self.draw_menu()
            else:
                self.screen.fill(BOARD_BG)
                self.hex_board.draw(self.board)
                self.draw_ui()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Chinese Checkers against AI (GUI)')
    parser.add_argument('--model', type=str, default='test_checkpoints/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--mcts-iterations', type=int, default=200,
                        help='MCTS iterations for AI (higher = stronger but slower)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run model on (cpu or cuda)')
    parser.add_argument('--mode', type=str, default='menu',
                        choices=['menu', 'human_vs_ai_p1', 'human_vs_ai_p2', 'human_vs_human'],
                        help='Game mode (default: menu shows selection screen)')
    
    args = parser.parse_args()
    
    game = GameGUI(
        model_path=args.model,
        mcts_iterations=args.mcts_iterations,
        device=args.device,
        game_mode=args.mode if args.mode != 'menu' else 'human_vs_ai_p1'
    )
    
    # If mode is menu, show menu; otherwise start directly
    if args.mode == 'menu':
        game.show_menu = True
    else:
        game.show_menu = False
    
    game.run()


if __name__ == '__main__':
    main()
