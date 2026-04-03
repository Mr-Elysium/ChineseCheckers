#!/usr/bin/env python3
"""
CLI Interface for Chinese Checkers
Simple terminal-based interface for playing against AI
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import torch
import numpy as np
import cc_core
from network import ChineseCheckersNet
from utils import hex_to_tensor, ROWS, COLS


class ChineseCheckersCLI:
    def __init__(self, model_path=None, mcts_iterations=200, device='cpu'):
        """Initialize the CLI game."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mcts_iterations = mcts_iterations
        
        # Initialize C++ engine
        cc_core.MoveGen.initialize()
        self.board = cc_core.Board(2)
        self.mcts = cc_core.MCTS(1.41)
        
        # Load AI model if provided
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
            print(f"✓ Loaded model from {model_path}")
            print(f"  Architecture: {config.get('num_res_blocks', 10)} blocks, {config.get('num_channels', 128)} channels")
        else:
            print("⚠ No model loaded - AI will use random policy")
    
    def display_board(self):
        """Display the board in a readable CLI format."""
        grid = self.board.get_grid()
        full_grid = np.zeros((17, 17), dtype=int)
        full_grid[ROWS, COLS] = grid
        
        print("\n" + "=" * 70)
        print("Chinese Checkers Board (17x17 grid, 121 valid positions)")
        print("=" * 70)
        print("Legend: . = empty, 1 = Player 1 (YOU), 2 = Player 2 (AI)")
        print()
        
        # Display with row/col indices
        print("     ", end="")
        for c in range(17):
            print(f"{c:2}", end=" ")
        print()
        
        for r in range(17):
            print(f" {r:2}: ", end="")
            for c in range(17):
                if full_grid[r, c] == 0 and (r, c) in zip(ROWS, COLS):
                    print(" .", end=" ")
                elif full_grid[r, c] == 1:
                    print(" 1", end=" ")
                elif full_grid[r, c] == 2:
                    print(" 2", end=" ")
                else:
                    print("  ", end=" ")
            print()
        print()
    
    def get_position_index(self, row, col):
        """Convert (row, col) to the 121-node index, or return -1 if invalid."""
        for i, (r, c) in enumerate(zip(ROWS, COLS)):
            if r == row and c == col:
                return i
        return -1
    
    def get_position_coords(self, idx):
        """Convert 121-node index to (row, col)."""
        if 0 <= idx < 121:
            return ROWS[idx], COLS[idx]
        return None, None
    
    def display_legal_moves(self):
        """Show all legal moves for the current player."""
        legal_moves = cc_core.MoveGen.get_legal_moves(self.board)
        
        if not legal_moves:
            print("No legal moves available!")
            return
        
        print(f"\nLegal moves for Player {self.board.get_current_player()}:")
        print("-" * 70)
        
        # Group by starting position
        move_dict = {}
        for move in legal_moves:
            from_coords = self.get_position_coords(move.from_idx)
            to_coords = self.get_position_coords(move.to_idx)
            
            if from_coords not in move_dict:
                move_dict[from_coords] = []
            move_dict[from_coords].append(to_coords)
        
        for from_pos, to_positions in sorted(move_dict.items()):
            print(f"From ({from_pos[0]:2},{from_pos[1]:2}) → ", end="")
            to_str = ", ".join([f"({t[0]:2},{t[1]:2})" for t in to_positions[:5]])
            if len(to_positions) > 5:
                to_str += f" ... +{len(to_positions)-5} more"
            print(to_str)
        
        print(f"\nTotal: {len(legal_moves)} legal moves")
    
    def get_human_move(self):
        """Get move input from human player."""
        legal_moves = cc_core.MoveGen.get_legal_moves(self.board)
        
        while True:
            print("\n" + "-" * 70)
            print("Enter your move:")
            print("  Format: 'from_row from_col to_row to_col' (e.g., '0 8 1 8')")
            print("  Commands: 'moves' (show legal moves) | 'quit' (exit game)")
            print("-" * 70)
            
            user_input = input("> ").strip().lower()
            
            if user_input == 'quit':
                return None
            
            if user_input == 'moves':
                self.display_legal_moves()
                continue
            
            try:
                parts = user_input.split()
                if len(parts) != 4:
                    print("❌ Invalid format. Need 4 numbers: from_row from_col to_row to_col")
                    continue
                
                from_r, from_c, to_r, to_c = map(int, parts)
                
                from_idx = self.get_position_index(from_r, from_c)
                to_idx = self.get_position_index(to_r, to_c)
                
                if from_idx == -1:
                    print(f"❌ ({from_r},{from_c}) is not a valid board position")
                    continue
                
                if to_idx == -1:
                    print(f"❌ ({to_r},{to_c}) is not a valid board position")
                    continue
                
                # Check if move is legal
                move_found = None
                for move in legal_moves:
                    if move.from_idx == from_idx and move.to_idx == to_idx:
                        move_found = move
                        break
                
                if move_found:
                    return move_found
                else:
                    print(f"❌ That move is not legal. Type 'moves' to see legal moves.")
                    
            except ValueError:
                print("❌ Invalid input. Please enter 4 numbers.")
    
    def get_ai_move(self):
        """Get move from AI using MCTS + neural network."""
        print(f"\n🤖 AI (Player {self.board.get_current_player()}) is thinking...")
        
        def predictor(board_array):
            """Neural network predictor for MCTS."""
            if self.model is None:
                # Random policy if no model
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
        
        from_coords = self.get_position_coords(best_move.from_idx)
        to_coords = self.get_position_coords(best_move.to_idx)
        print(f"🤖 AI moves from ({from_coords[0]},{from_coords[1]}) to ({to_coords[0]},{to_coords[1]})")
        
        return best_move
    
    def play(self, human_player=1):
        """Main game loop."""
        print("\n" + "=" * 70)
        print("CHINESE CHECKERS - Human vs AI")
        print("=" * 70)
        print(f"You are Player {human_player}")
        print(f"AI is Player {3 - human_player}")
        print(f"AI using {self.mcts_iterations} MCTS iterations per move")
        print("=" * 70)
        
        move_count = 0
        max_moves = 400
        
        while not self.board.is_terminal() and move_count < max_moves:
            self.display_board()
            
            current_player = self.board.get_current_player()
            print(f"\n>>> Move {move_count + 1} - Player {current_player}'s turn <<<")
            
            if current_player == human_player:
                # Human turn
                move = self.get_human_move()
                if move is None:
                    print("\n👋 Game ended by user")
                    return
            else:
                # AI turn
                move = self.get_ai_move()
            
            self.board.apply_move(move)
            move_count += 1
            
            if move_count % 50 == 0:
                print(f"\n⏱ Move {move_count}/{max_moves}")
        
        # Game over
        self.display_board()
        
        if self.board.is_terminal():
            rewards = self.board.get_rewards()
            if rewards[0] > rewards[1]:
                winner = 1
            else:
                winner = 2
            
            print("\n" + "=" * 70)
            print("🎉 GAME OVER!")
            print("=" * 70)
            print(f"Winner: Player {winner}")
            
            if winner == human_player:
                print("🏆 Congratulations! You beat the AI!")
            else:
                print("😔 The AI won this time. Try again!")
        else:
            print("\n" + "=" * 70)
            print("⏱ GAME OVER - Maximum moves reached (Draw)")
            print("=" * 70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Play Chinese Checkers against AI (CLI)')
    parser.add_argument('--model', type=str, default='test_checkpoints/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--mcts-iterations', type=int, default=200,
                        help='MCTS iterations for AI (higher = stronger but slower)')
    parser.add_argument('--human-player', type=int, default=1, choices=[1, 2],
                        help='Which player you want to be (1 or 2)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run model on (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"⚠ Warning: Model file not found at {args.model}")
        print("AI will use random policy. Train a model first!")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    game = ChineseCheckersCLI(
        model_path=args.model,
        mcts_iterations=args.mcts_iterations,
        device=args.device
    )
    
    game.play(human_player=args.human_player)


if __name__ == '__main__':
    main()
