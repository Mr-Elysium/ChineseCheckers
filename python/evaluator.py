import torch
import numpy as np
import cc_core
from utils import hex_to_tensor
from tqdm import tqdm

class Evaluator:
    def __init__(self, num_games=20, mcts_iterations=100, device='cuda', verbose=False, use_progress_bar=True):
        self.num_games = num_games
        self.mcts_iterations = mcts_iterations
        
        # Handle device selection - accept device object or string
        if isinstance(device, torch.device):
            self.device = device
        elif device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif device == 'mps' and torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        self.verbose = verbose
        self.use_progress_bar = use_progress_bar
        cc_core.MoveGen.initialize()
    
    def evaluate(self, model_new, model_old):
        model_new.eval()
        model_old.eval()
        
        wins_new = 0
        wins_old = 0
        draws = 0
        
        # Use tqdm if enabled
        game_iter = tqdm(range(self.num_games), desc="  Evaluation", unit="game") if self.use_progress_bar else range(self.num_games)
        
        for game_idx in game_iter:
            if game_idx % 2 == 0:
                result = self._play_game(model_new, model_old)
                if result == 1:
                    wins_new += 1
                elif result == -1:
                    wins_old += 1
                else:
                    draws += 1
            else:
                result = self._play_game(model_old, model_new)
                if result == 1:
                    wins_old += 1
                elif result == -1:
                    wins_new += 1
                else:
                    draws += 1
            
            if not self.use_progress_bar and self.verbose and (game_idx + 1) % 5 == 0:
                print(f"  Evaluation: {game_idx + 1}/{self.num_games} games complete")
        
        win_rate = wins_new / self.num_games
        print(f"\nEvaluation Results:")
        print(f"  New Model: {wins_new} wins ({win_rate:.1%})")
        print(f"  Old Model: {wins_old} wins ({(wins_old/self.num_games):.1%})")
        print(f"  Draws:     {draws} ({(draws/self.num_games):.1%})")
        
        return win_rate
    
    def _play_game(self, model_p1, model_p2):
        board = cc_core.Board(2)
        mcts_p1 = cc_core.MCTS(1.41)
        mcts_p2 = cc_core.MCTS(1.41)
        
        move_count = 0
        max_moves = 400
        
        while not board.is_terminal() and move_count < max_moves:
            current_player = board.get_current_player()
            
            if current_player == 1:
                predictor = self._make_predictor(model_p1, board)
                mcts_p1.search(self.mcts_iterations, board, predictor)
                best_move = mcts_p1.get_best_move()
            else:
                predictor = self._make_predictor(model_p2, board)
                mcts_p2.search(self.mcts_iterations, board, predictor)
                best_move = mcts_p2.get_best_move()
            
            board.apply_move(best_move)
            move_count += 1
        
        if board.is_terminal():
            rewards = board.get_rewards()
            if rewards[0] > rewards[1]:
                return 1
            elif rewards[1] > rewards[0]:
                return -1
        
        return 0
    
    def _make_predictor(self, model, board):
        def predictor(board_array):
            current_player = board.get_current_player()
            state_tensor = hex_to_tensor(np.array(board_array), current_player)
            
            with torch.no_grad():
                state_tensor = state_tensor.unsqueeze(0).to(self.device)
                policy_logits, value = model(state_tensor)
                
                policy = torch.softmax(policy_logits, dim=1)[0].cpu().numpy()
                value_vec = value[0].cpu().numpy()
            
            if len(value_vec) == 1:
                value_output = [value_vec[0], -value_vec[0]]
            else:
                value_output = value_vec.tolist()
            
            return policy.tolist(), value_output
        
        return predictor

if __name__ == "__main__":
    print("=== Testing evaluator.py ===\n")
    
    # Test 1: Evaluator initialization
    print("Test 1: Evaluator Initialization...")
    evaluator = Evaluator(num_games=2, mcts_iterations=10, device='cpu')
    assert evaluator.num_games == 2, "Num games mismatch"
    assert evaluator.mcts_iterations == 10, "MCTS iterations mismatch"
    print("✓ Evaluator initialized correctly\n")
    
    # Test 2: Predictor creation
    print("Test 2: Predictor Creation...")
    from network import ChineseCheckersNet
    
    model = ChineseCheckersNet(num_players=2)
    model.eval()
    
    board = cc_core.Board(2)
    predictor = evaluator._make_predictor(model, board)
    
    grid = board.get_grid()
    policy, value = predictor(grid.tolist())
    
    assert len(policy) == 121, f"Policy should be 121 elements, got {len(policy)}"
    assert len(value) == 2, f"Value should be 2 elements, got {len(value)}"
    assert abs(sum(policy) - 1.0) < 1e-5, "Policy should sum to 1.0"
    print("✓ Predictor works correctly\n")
    
    # Test 3: Single game play
    print("Test 3: Single Game Play...")
    result = evaluator._play_game(model, model)
    assert result in [-1, 0, 1], f"Result should be -1, 0, or 1, got {result}"
    print(f"✓ Game completed with result: {result}\n")
    
    # Test 4: Full evaluation (quick)
    print("Test 4: Full Evaluation...")
    win_rate = evaluator.evaluate(model, model)
    assert 0.0 <= win_rate <= 1.0, f"Win rate should be in [0, 1], got {win_rate}"
    print(f"✓ Evaluation completed (win rate: {win_rate:.1%})\n")
    
    print("=" * 50)
    print("ALL EVALUATOR TESTS PASSED (4/4)")
    print("=" * 50)
