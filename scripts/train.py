#!/usr/bin/env python3
import sys
import os
import argparse
import multiprocessing as mp
import yaml

mp.set_start_method('spawn', force=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from trainer import Trainer
import torch

def load_config_file(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested config structure
    flat_config = {}
    if 'model' in config:
        flat_config.update(config['model'])
    if 'training' in config:
        flat_config.update(config['training'])
    if 'self_play' in config:
        flat_config.update(config['self_play'])
    if 'performance' in config:
        flat_config.update(config['performance'])
    if 'replay_buffer' in config:
        flat_config.update(config['replay_buffer'])
    if 'evaluation' in config:
        flat_config.update(config['evaluation'])
    if 'training_loop' in config:
        flat_config.update(config['training_loop'])
    if 'directories' in config:
        flat_config.update(config['directories'])
    if 'wandb' in config:
        flat_config['wandb_enabled'] = config['wandb'].get('enabled', False)
        flat_config['wandb_project'] = config['wandb'].get('project', 'chinese-checkers-alphazero')
        flat_config['wandb_run_name'] = config['wandb'].get('run_name', None)
    if 'ui' in config:
        flat_config.update(config['ui'])
    
    return flat_config

def parse_args():
    parser = argparse.ArgumentParser(description='Train Chinese Checkers AlphaZero')
    
    # Config file
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to YAML config file')
    
    # Model architecture
    parser.add_argument('--num-players', type=int, default=2, help='Number of players (2 or 6)')
    parser.add_argument('--num-res-blocks', type=int, default=10, help='Number of residual blocks')
    parser.add_argument('--num-channels', type=int, default=128, help='Number of channels in ResNet')
    
    # Training hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--batch-size', type=int, default=512, help='Training batch size')
    parser.add_argument('--training-steps', type=int, default=100, help='Training steps per iteration')
    
    # Self-play settings
    parser.add_argument('--games-per-iteration', type=int, default=200, help='Self-play games per iteration')
    parser.add_argument('--num-workers', type=int, default=16, help='Number of self-play workers')
    parser.add_argument('--mcts-iterations', type=int, default=100, help='MCTS iterations per move')
    parser.add_argument('--inference-batch-size', type=int, default=None, help='GPU inference batch size (default: auto-detect based on device)')
    parser.add_argument('--batch-delay', type=float, default=None, help='Batch collection delay in seconds (default: auto-detect based on device)')
    
    # Performance settings
    parser.add_argument('--num-threads', type=int, default=None, help='PyTorch thread count (default: auto-detect CPU cores)')
    
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
    
    # Load config from YAML file (if exists)
    yaml_config = load_config_file(args.config)
    
    # Helper function to get value: CLI arg > YAML config > default
    def get_value(arg_value, yaml_key, default_value):
        # If CLI argument was explicitly provided (not default), use it
        if arg_value != default_value:
            return arg_value
        # Otherwise, use YAML config if available
        return yaml_config.get(yaml_key, default_value)
    
    # Auto-detect optimal settings based on hardware
    cpu_count = mp.cpu_count()
    
    # Get parser defaults for comparison
    parser_defaults = {
        'num_players': 2,
        'num_res_blocks': 10,
        'num_channels': 128,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'batch_size': 512,
        'training_steps': 100,
        'games_per_iteration': 200,
        'num_workers': 16,
        'mcts_iterations': 100,
        'buffer_size': 100000,
        'eval_games': 40,
        'eval_frequency': 5,
        'win_rate_threshold': 0.55,
        'num_iterations': 1000,
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'logs',
        'wandb_project': 'chinese-checkers-alphazero'
    }
    
    # Merge config: YAML < CLI args
    num_players = get_value(args.num_players, 'num_players', parser_defaults['num_players'])
    num_res_blocks = get_value(args.num_res_blocks, 'num_res_blocks', parser_defaults['num_res_blocks'])
    num_channels = get_value(args.num_channels, 'num_channels', parser_defaults['num_channels'])
    learning_rate = get_value(args.learning_rate, 'learning_rate', parser_defaults['learning_rate'])
    weight_decay = get_value(args.weight_decay, 'weight_decay', parser_defaults['weight_decay'])
    batch_size = get_value(args.batch_size, 'batch_size', parser_defaults['batch_size'])
    training_steps = get_value(args.training_steps, 'training_steps', parser_defaults['training_steps'])
    games_per_iteration = get_value(args.games_per_iteration, 'games_per_iteration', parser_defaults['games_per_iteration'])
    num_workers = get_value(args.num_workers, 'num_workers', parser_defaults['num_workers'])
    mcts_iterations = get_value(args.mcts_iterations, 'mcts_iterations', parser_defaults['mcts_iterations'])
    buffer_size = get_value(args.buffer_size, 'buffer_size', parser_defaults['buffer_size'])
    eval_games = get_value(args.eval_games, 'eval_games', parser_defaults['eval_games'])
    eval_frequency = get_value(args.eval_frequency, 'eval_frequency', parser_defaults['eval_frequency'])
    win_rate_threshold = get_value(args.win_rate_threshold, 'win_rate_threshold', parser_defaults['win_rate_threshold'])
    num_iterations = get_value(args.num_iterations, 'num_iterations', parser_defaults['num_iterations'])
    checkpoint_dir = get_value(args.checkpoint_dir, 'checkpoint_dir', parser_defaults['checkpoint_dir'])
    log_dir = get_value(args.log_dir, 'log_dir', parser_defaults['log_dir'])
    wandb_project = get_value(args.wandb_project, 'wandb_project', parser_defaults['wandb_project'])
    
    # Auto-detect thread count (CLI > YAML > auto-detect)
    if args.num_threads is not None:
        num_threads = args.num_threads
    elif 'num_threads' in yaml_config:
        num_threads = yaml_config['num_threads']
    else:
        num_threads = cpu_count
    
    torch.set_num_threads(num_threads)
    torch.set_float32_matmul_precision('high')
    
    # Auto-detect device for inference batch size and delay optimization
    if torch.cuda.is_available():
        device_type = 'cuda'
        default_inference_batch = 128
        default_batch_delay = 0.005
    elif torch.backends.mps.is_available():
        device_type = 'mps'
        default_inference_batch = 32
        default_batch_delay = 0.010
    else:
        device_type = 'cpu'
        default_inference_batch = 16
        default_batch_delay = 0.015
    
    # Inference batch size: CLI > YAML > auto-detect
    if args.inference_batch_size is not None:
        inference_batch_size = args.inference_batch_size
    elif 'inference_batch_size' in yaml_config:
        inference_batch_size = yaml_config['inference_batch_size']
    else:
        inference_batch_size = default_inference_batch
    
    # Batch delay: CLI > YAML > auto-detect
    if args.batch_delay is not None:
        batch_delay = args.batch_delay
    elif 'batch_delay' in yaml_config:
        batch_delay = yaml_config['batch_delay']
    else:
        batch_delay = default_batch_delay
    
    # W&B settings
    use_wandb = args.wandb or yaml_config.get('wandb_enabled', False)
    wandb_run_name = args.wandb_run_name or yaml_config.get('wandb_run_name', None)
    
    # UI settings
    verbose = args.verbose or yaml_config.get('verbose', False)
    use_progress_bar = not args.no_progress_bar and yaml_config.get('progress_bar', True)
    
    print(f"Configuration loaded from: {args.config if os.path.exists(args.config) else 'defaults (no config file)'}")
    print(f"Auto-detected settings:")
    print(f"  Device: {device_type}")
    print(f"  CPU cores: {cpu_count}")
    print(f"  PyTorch threads: {num_threads}")
    print(f"  Inference batch size: {inference_batch_size}")
    print(f"  Batch delay: {batch_delay*1000:.1f}ms")
    
    # Build config dictionary
    config = {
        'num_players': num_players,
        'num_res_blocks': num_res_blocks,
        'num_channels': num_channels,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'buffer_size': buffer_size,
        'batch_size': batch_size,
        'training_steps': training_steps,
        'games_per_iteration': games_per_iteration,
        'num_workers': num_workers,
        'mcts_iterations': mcts_iterations,
        'inference_batch_size': inference_batch_size,
        'batch_delay': batch_delay,
        'eval_games': eval_games,
        'eval_frequency': eval_frequency,
        'win_rate_threshold': win_rate_threshold,
        'num_iterations': num_iterations,
        'checkpoint_dir': checkpoint_dir,
        'log_dir': log_dir,
        'verbose': verbose,
        'use_wandb': use_wandb,
        'wandb_project': wandb_project,
        'wandb_run_name': wandb_run_name,
        'use_progress_bar': use_progress_bar
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
