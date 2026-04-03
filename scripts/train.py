#!/usr/bin/env python3
import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train Chinese Checkers AlphaZero')
    
    # Model architecture
    parser.add_argument('--num-players', type=int, default=2, help='Number of players (2 or 6)')
    parser.add_argument('--num-res-blocks', type=int, default=10, help='Number of residual blocks')
    parser.add_argument('--num-channels', type=int, default=128, help='Number of channels in ResNet')
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--training-steps', type=int, default=100, help='Training steps per iteration')
    
    # Self-play settings
    parser.add_argument('--games-per-iteration', type=int, default=200, help='Self-play games per iteration')
    parser.add_argument('--num-workers', type=int, default=18, help='Number of self-play workers')
    parser.add_argument('--mcts-iterations', type=int, default=200, help='MCTS iterations per move')
    
    # Replay buffer
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    
    # Evaluation
    parser.add_argument('--eval-games', type=int, default=40, help='Evaluation games')
    parser.add_argument('--eval-frequency', type=int, default=5, help='Evaluate every N iterations')
    parser.add_argument('--win-rate-threshold', type=float, default=0.55, help='Win rate to replace best model')
    
    # Training loop
    parser.add_argument('--num-iterations', type=int, default=1000, help='Total training iterations')
    
    # Directories
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='logs', help='Log directory')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # Verbosity
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging (move-by-move details)')
    
    # Experiment tracking
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases experiment tracking')
    parser.add_argument('--wandb-project', type=str, default='chinese-checkers-alphazero', help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name (auto-generated if not set)')
    parser.add_argument('--no-progress-bar', action='store_true', help='Disable tqdm progress bars')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Build config dictionary
    config = {
        'num_players': args.num_players,
        'num_res_blocks': args.num_res_blocks,
        'num_channels': args.num_channels,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'buffer_size': args.buffer_size,
        'batch_size': args.batch_size,
        'training_steps': args.training_steps,
        'games_per_iteration': args.games_per_iteration,
        'num_workers': args.num_workers,
        'mcts_iterations': args.mcts_iterations,
        'eval_games': args.eval_games,
        'eval_frequency': args.eval_frequency,
        'win_rate_threshold': args.win_rate_threshold,
        'num_iterations': args.num_iterations,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'verbose': args.verbose,
        'use_wandb': args.wandb,
        'wandb_project': args.wandb_project,
        'wandb_run_name': args.wandb_run_name,
        'use_progress_bar': not args.no_progress_bar
    }
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer._save_checkpoint(is_best=False)
        print("Checkpoint saved. You can resume with --resume")

if __name__ == '__main__':
    main()
