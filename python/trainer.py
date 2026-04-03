import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import multiprocessing as mp
from queue import Empty
from datetime import datetime

from network import ChineseCheckersNet
from replay_buffer import ReplayBuffer
from evaluator import Evaluator
from self_play import gpu_inference_server, actor_worker
from utils import hex_to_tensor

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize network
        self.model = ChineseCheckersNet(
            num_players=config['num_players'],
            num_res_blocks=config['num_res_blocks'],
            num_channels=config['num_channels']
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(max_size=config['buffer_size'])
        
        # Evaluator
        self.evaluator = Evaluator(
            num_games=config['eval_games'],
            mcts_iterations=config['mcts_iterations'],
            device=self.device,
            verbose=config.get('verbose', False)
        )
        
        # Training state
        self.iteration = 0
        self.best_model_path = None
        
        # Create directories
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
    
    def train(self):
        print("=" * 60)
        print("Starting AlphaZero Training")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Replay buffer size: {self.config['buffer_size']}")
        print(f"Self-play games per iteration: {self.config['games_per_iteration']}")
        print("=" * 60)
        
        for iteration in range(self.config['num_iterations']):
            self.iteration = iteration
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{self.config['num_iterations']}")
            print(f"{'='*60}")
            
            # Phase 1: Self-play
            print("\n[1/3] Collecting self-play data...")
            start_time = time.time()
            trajectories = self._collect_self_play_data()
            
            for traj in trajectories:
                self.replay_buffer.add_trajectory(traj)
            
            elapsed = time.time() - start_time
            print(f"  Collected {len(trajectories)} games in {elapsed:.1f}s")
            print(f"  Replay buffer size: {len(self.replay_buffer)}")
            
            # Phase 2: Training
            print("\n[2/3] Training network...")
            start_time = time.time()
            train_stats = self._train_network()
            elapsed = time.time() - start_time
            
            print(f"  Training completed in {elapsed:.1f}s")
            print(f"  Policy loss: {train_stats['policy_loss']:.4f}")
            print(f"  Value loss:  {train_stats['value_loss']:.4f}")
            print(f"  Total loss:  {train_stats['total_loss']:.4f}")
            
            # Phase 3: Evaluation
            if (iteration + 1) % self.config['eval_frequency'] == 0:
                print("\n[3/3] Evaluating model...")
                self._evaluate_and_save()
            else:
                print("\n[3/3] Skipping evaluation (not due yet)")
                self._save_checkpoint(is_best=False)
            
            # Log iteration summary
            self._log_iteration(iteration, train_stats, len(trajectories))
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
    
    def _collect_self_play_data(self):
        num_workers = self.config['num_workers']
        games_per_worker = self.config['games_per_iteration'] // num_workers
        
        # Save current model weights to temporary file for GPU server
        import tempfile
        temp_model = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pt')
        torch.save(self.model.state_dict(), temp_model.name)
        temp_model.close()
        
        # Shared queues and pipes
        task_queue = mp.Queue()
        game_output_queue = mp.Queue()
        pipes = [mp.Pipe() for _ in range(num_workers)]
        
        # Start GPU inference server with current model weights
        verbose = self.config.get('verbose', False)
        gpu_process = mp.Process(
            target=gpu_inference_server,
            args=(task_queue, [p[1] for p in pipes], temp_model.name, 
                  self.config['num_players'], 64, 
                  self.config['num_res_blocks'], self.config['num_channels'], verbose)
        )
        gpu_process.start()
        
        # Start actor workers
        workers = []
        for i in range(num_workers):
            p = mp.Process(
                target=actor_worker,
                args=(i, task_queue, pipes[i][0], game_output_queue, self.config['mcts_iterations'], verbose)
            )
            p.start()
            workers.append(p)
        
        # Collect games
        trajectories = []
        games_collected = 0
        target_games = self.config['games_per_iteration']
        
        while games_collected < target_games:
            try:
                trajectory = game_output_queue.get(timeout=1.0)
                trajectories.append(trajectory)
                games_collected += 1
                
                if self.config.get('verbose', False) and games_collected % 10 == 0:
                    print(f"  Progress: {games_collected}/{target_games} games")
                elif games_collected % 25 == 0:
                    print(f"  Progress: {games_collected}/{target_games} games")
            except Empty:
                continue
        
        # Cleanup
        for p in workers:
            p.terminate()
            p.join()
        
        gpu_process.terminate()
        gpu_process.join()
        
        # Remove temporary model file
        os.unlink(temp_model.name)
        
        return trajectories
    
    def _train_network(self):
        self.model.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for _ in range(self.config['training_steps']):
            # Sample batch
            states, policies, rewards, players = self.replay_buffer.sample_batch(
                self.config['batch_size']
            )
            
            # Convert to tensors
            batch_tensors = []
            for i in range(len(states)):
                tensor = hex_to_tensor(states[i], players[i])
                batch_tensors.append(tensor)
            
            input_batch = torch.stack(batch_tensors).to(self.device)
            target_policies = torch.from_numpy(policies).to(self.device)
            target_values = torch.from_numpy(rewards).to(self.device)
            
            # Forward pass
            pred_policies, pred_values = self.model(input_batch)
            
            # Compute losses
            # Policy loss: KL divergence between MCTS policy and network policy
            log_probs = F.log_softmax(pred_policies, dim=1)
            policy_loss = F.kl_div(
                log_probs,
                target_policies,
                reduction='batchmean'
            )
            
            # Value loss: MSE for current player's reward
            if self.config['num_players'] == 2:
                # For 2-player, use player's actual reward
                player_rewards = torch.zeros(len(players), 1).to(self.device)
                for i, player in enumerate(players):
                    player_rewards[i] = target_values[i, player - 1]
                value_loss = F.mse_loss(pred_values, player_rewards)
            else:
                # For 6-player, use full reward vector
                value_loss = F.mse_loss(pred_values, target_values)
            
            # Total loss
            loss = policy_loss + value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track stats
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'total_loss': total_loss / num_batches
        }
    
    def _evaluate_and_save(self):
        if self.best_model_path is None:
            # First evaluation, save as best
            print("  First model - saving as best")
            self._save_checkpoint(is_best=True)
            return
        
        # Load best model for comparison
        best_model = ChineseCheckersNet(
            num_players=self.config['num_players'],
            num_res_blocks=self.config['num_res_blocks'],
            num_channels=self.config['num_channels']
        ).to(self.device)
        
        checkpoint = torch.load(self.best_model_path)
        best_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate
        win_rate = self.evaluator.evaluate(self.model, best_model)
        
        # Update best model if win rate > threshold
        if win_rate >= self.config['win_rate_threshold']:
            print(f"  New model wins {win_rate:.1%} - Updating best model!")
            self._save_checkpoint(is_best=True)
        else:
            print(f"  New model wins {win_rate:.1%} - Keeping old model")
            self._save_checkpoint(is_best=False)
    
    def _save_checkpoint(self, is_best=False):
        checkpoint = {
            'iteration': self.iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(
            self.config['checkpoint_dir'],
            f'checkpoint_iter_{self.iteration}.pt'
        )
        torch.save(checkpoint, latest_path)
        
        # Save as best if applicable
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint_dir'],
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            print(f"  Saved best model: {best_path}")
    
    def _log_iteration(self, iteration, train_stats, num_games):
        log_entry = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'num_games': num_games,
            'buffer_size': len(self.replay_buffer),
            'policy_loss': train_stats['policy_loss'],
            'value_loss': train_stats['value_loss'],
            'total_loss': train_stats['total_loss']
        }
        
        log_file = os.path.join(self.config['log_dir'], 'training_log.jsonl')
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        print(f"Loaded checkpoint from iteration {self.iteration}")

if __name__ == "__main__":
    print("=== Testing trainer.py ===\n")
    
    # Test configuration
    test_config = {
        'num_players': 2,
        'num_res_blocks': 5,
        'num_channels': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'buffer_size': 1000,
        'batch_size': 32,
        'training_steps': 10,
        'games_per_iteration': 4,
        'num_workers': 2,
        'mcts_iterations': 20,
        'eval_games': 2,
        'eval_frequency': 5,
        'win_rate_threshold': 0.55,
        'num_iterations': 1,
        'checkpoint_dir': 'test_checkpoints',
        'log_dir': 'test_logs'
    }
    
    # Test 1: Trainer initialization
    print("Test 1: Trainer Initialization...")
    trainer = Trainer(test_config)
    assert trainer.iteration == 0, "Iteration should start at 0"
    assert len(trainer.replay_buffer) == 0, "Buffer should start empty"
    print("✓ Trainer initialized correctly\n")
    
    # Test 2: Checkpoint save/load
    print("Test 2: Checkpoint Save/Load...")
    trainer._save_checkpoint(is_best=True)
    
    new_trainer = Trainer(test_config)
    checkpoint_path = os.path.join(test_config['checkpoint_dir'], 'best_model.pt')
    new_trainer.load_checkpoint(checkpoint_path)
    print("✓ Checkpoint save/load works\n")
    
    # Test 3: Training step (with dummy data)
    print("Test 3: Training Step...")
    # Add dummy data to buffer
    for _ in range(100):
        dummy_step = {
            'state': np.random.rand(121),
            'player': np.random.randint(1, 3),
            'policy': np.random.rand(121),
            'rewards': [1.0, -1.0]
        }
        dummy_step['policy'] /= dummy_step['policy'].sum()
        trainer.replay_buffer.buffer.append(dummy_step)
    
    stats = trainer._train_network()
    assert 'policy_loss' in stats, "Stats should contain policy_loss"
    assert 'value_loss' in stats, "Stats should contain value_loss"
    assert 'total_loss' in stats, "Stats should contain total_loss"
    print(f"✓ Training step completed")
    print(f"  Policy loss: {stats['policy_loss']:.4f}")
    print(f"  Value loss:  {stats['value_loss']:.4f}\n")
    
    # Cleanup test files
    import shutil
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')
    if os.path.exists('test_logs'):
        shutil.rmtree('test_logs')
    
    print("=" * 50)
    print("ALL TRAINER TESTS PASSED (3/3)")
    print("=" * 50)
    print("\nNote: Full training requires running trainer.train()")
    print("which orchestrates self-play, training, and evaluation.")
