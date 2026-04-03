#!/usr/bin/env python3
"""
Chinese Checkers - Human vs AI
Main entry point with CLI and GUI options
"""

import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='Play Chinese Checkers against your trained AlphaZero AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Play with GUI (requires pygame)
  python ui/play.py --gui
  
  # Play in terminal
  python ui/play.py
  
  # Use specific model and difficulty
  python ui/play.py --gui --model checkpoints/best_model.pt --mcts-iterations 400
  
  # Play as Player 2
  python ui/play.py --human-player 2
        """
    )
    
    parser.add_argument('--gui', action='store_true',
                        help='Use Pygame GUI interface (default: CLI)')
    parser.add_argument('--model', type=str, default='test_checkpoints/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--mcts-iterations', type=int, default=200,
                        help='MCTS iterations for AI (higher = stronger but slower)')
    parser.add_argument('--human-player', type=int, default=1, choices=[1, 2],
                        help='Which player you want to be (1 or 2) - CLI only')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run model on (cpu or cuda)')
    parser.add_argument('--mode', type=str, default='menu',
                        choices=['menu', 'human_vs_ai_p1', 'human_vs_ai_p2', 'human_vs_human'],
                        help='Game mode for GUI (default: menu shows selection screen)')
    
    args = parser.parse_args()
    
    if args.gui:
        # Try to use GUI
        try:
            from play_gui import GameGUI
            print("Starting GUI mode...")
            
            # Determine game mode
            if args.mode == 'menu':
                game_mode = 'human_vs_ai_p1'
                show_menu = True
            else:
                game_mode = args.mode
                show_menu = False
            
            game = GameGUI(
                model_path=args.model,
                mcts_iterations=args.mcts_iterations,
                device=args.device,
                game_mode=game_mode
            )
            game.show_menu = show_menu
            game.run()
        except ImportError as e:
            print(f"❌ Error: {e}")
            print("\nPygame not installed. Install with:")
            print("  pip install pygame")
            print("\nFalling back to CLI mode...\n")
            
            from play_cli import ChineseCheckersCLI
            game = ChineseCheckersCLI(
                model_path=args.model,
                mcts_iterations=args.mcts_iterations,
                device=args.device
            )
            game.play(human_player=args.human_player)
    else:
        # Use CLI
        from play_cli import ChineseCheckersCLI
        print("Starting CLI mode...")
        game = ChineseCheckersCLI(
            model_path=args.model,
            mcts_iterations=args.mcts_iterations,
            device=args.device
        )
        game.play(human_player=args.human_player)


if __name__ == '__main__':
    main()
