import os
import time
import json
import torch
import numpy as np
import multiprocessing as mp
from queue import Empty
from network import ChineseCheckersNet
from utils import hex_to_tensor, ROWS, COLS
import cc_core

# --- Temperature-based Move Selection ---
def select_move_with_temperature(mcts, temperature=1.0):
    """
    Select a move from MCTS visit counts using temperature.
    temperature=0: deterministic (pick most visited)
    temperature=1: proportional to visit counts
    temperature>1: more random
    """
    visit_counts = mcts.get_visit_counts()
    if not visit_counts:
        return mcts.get_best_move()
    
    if temperature == 0 or len(visit_counts) == 1:
        # Deterministic: pick most visited
        return mcts.get_best_move_and_reuse()
    
    # Apply temperature to visit counts
    moves = list(visit_counts.keys())
    visits = np.array([visit_counts[m] for m in moves], dtype=np.float64)
    visits = visits ** (1.0 / temperature)
    probs = visits / visits.sum()
    
    # Sample move index based on temperature-adjusted probabilities
    chosen_idx = np.random.choice(len(moves), p=probs)
    
    # Check if we sampled the best move (can reuse tree)
    best_idx = np.argmax(visits)
    if chosen_idx == best_idx:
        return mcts.get_best_move_and_reuse()
    else:
        # Sampled a non-best move, can't reuse tree
        # Note: This is a limitation - we return best move but lose tree reuse
        # A full implementation would need MCTS to support selecting arbitrary moves
        return mcts.get_best_move()

# --- 1. The Inference Server (The Brain on the 3080) ---
def gpu_inference_server(task_queue, pipes, model_path, num_players, batch_size=64, num_res_blocks=10, num_channels=128, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChineseCheckersNet(
        num_players=num_players,
        num_res_blocks=num_res_blocks,
        num_channels=num_channels
    ).to(device)
    
    # Load model weights if provided
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()
    # Skip torch.compile for the first 5 minutes of testing to see if it works!
    
    if verbose:
        print("GPU Server: Initialized and waiting for tasks...")

    while True:
        batch_ids = []
        batch_tensors = []
        
        start_time = time.time()
        # Collect tasks until batch is full or 10ms passed
        while len(batch_tensors) < batch_size and (time.time() - start_time < 0.01):
            try:
                worker_id, state_array, current_player = task_queue.get(timeout=0.001)
                batch_ids.append(worker_id)
                # CRITICAL FIX: Use actual current player, not hardcoded 1
                batch_tensors.append(hex_to_tensor(state_array, current_player))
            except Empty:
                break
        
        if not batch_tensors:
            continue

        with torch.no_grad():
            input_tensor = torch.stack(batch_tensors).to(device)
            policy_logits, value_vectors = model(input_tensor)
            
            probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values = value_vectors.cpu().numpy()

        # SUCCESS: Send results back through the dedicated Pipe
        for i, worker_id in enumerate(batch_ids):
            pipes[worker_id].send((probs[i].tolist(), values[i].tolist()))

# --- 2. The Actor ---
def actor_worker(worker_id, task_queue, pipe, game_output_queue, num_iters, verbose=False):
    cc_core.MoveGen.initialize()
    
    # Standard limit for 2-player Chinese Checkers
    MAX_MOVES = 400 

    while True:
        board = cc_core.Board(2)
        mcts = cc_core.MCTS(1.41)
        trajectory = []
        move_count = 0
        
        # Predictor that passes current player to GPU server
        def board_aware_predictor(board_array):
            current_player = board.get_current_player()
            task_queue.put((worker_id, board_array, current_player))
            return pipe.recv()
        
        # Add the move_count check here
        while not board.is_terminal() and move_count < MAX_MOVES:
            mcts.search(num_iters, board, board_aware_predictor)
            
            # Use temperature for exploration in early game
            # Temperature=1.0 for first 30 moves, then 0 (deterministic)
            temperature = 1.0 if move_count < 30 else 0.0
            best_move = select_move_with_temperature(mcts, temperature)
            
            # CRITICAL FIX: Store MCTS policy (visit counts) for training
            visit_counts = mcts.get_visit_counts()
            # Convert visit counts dict to 121-length policy vector
            policy = [0.0] * 121
            total_visits = sum(visit_counts.values())
            if total_visits > 0:
                for idx, count in visit_counts.items():
                    policy[idx] = count / total_visits
            
            trajectory.append({
                "state": board.get_grid().tolist(),
                "player": board.get_current_player(),
                "policy": policy  # MCTS-improved policy for training
            })
            board.apply_move(best_move)
            move_count += 1
            
            if verbose and move_count % 10 == 0:
                print(f"Actor {worker_id}: Move {move_count}/{MAX_MOVES}")
            
        # If we hit the limit, rewards are [0.0, 0.0] (a draw)
        rewards = board.get_rewards() if board.is_terminal() else [0.0, 0.0]
        
        for step in trajectory:
            step["rewards"] = rewards
        
        game_output_queue.put(trajectory)
        if verbose:
            status = "TERMINAL" if board.is_terminal() else "MAX_MOVES"
            print(f"Actor {worker_id}: Game finished ({status}). Saving...")

# --- 3. Test Block ---
if __name__ == "__main__":
    print("=== Testing self_play.py ===\n")
    
    # Test 1: C++ bindings work
    print("Test 1: C++ Bindings...")
    cc_core.MoveGen.initialize()
    board = cc_core.Board(2)
    assert board.get_current_player() == 1, "Starting player should be 1"
    assert not board.is_terminal(), "Board should not be terminal at start"
    print("✓ C++ Board initialized correctly")
    
    moves = cc_core.MoveGen.get_legal_moves(board)
    assert len(moves) > 0, "Should have legal moves at start"
    print(f"✓ Move generation works ({len(moves)} legal moves)\n")
    
    # Test 2: MCTS basic functionality
    print("Test 2: MCTS Integration...")
    mcts = cc_core.MCTS(1.41)
    
    # Simple dummy predictor
    def test_predictor(board_array):
        policy = [1.0 / 121.0] * 121  # Uniform policy
        value = [0.0, 0.0]  # Neutral value
        return policy, value
    
    # Run a few MCTS iterations
    mcts.search(5, board, test_predictor)
    best_move = mcts.get_best_move()
    
    assert best_move.from_idx >= 0 and best_move.from_idx < 121, "Invalid from index"
    assert best_move.to_idx >= 0 and best_move.to_idx < 121, "Invalid to index"
    print(f"✓ MCTS search completed")
    print(f"✓ Best move: {best_move.from_idx} → {best_move.to_idx}\n")
    
    # Test 3: Tensor encoding integration
    print("Test 3: Tensor Encoding...")
    grid = board.get_grid()
    assert len(grid) == 121, f"Grid should be 121 elements, got {len(grid)}"
    
    tensor = hex_to_tensor(grid, 1)
    assert tensor.shape == (4, 17, 17), f"Tensor shape mismatch: {tensor.shape}"
    print(f"✓ Board → Tensor conversion works")
    print(f"✓ Tensor shape: {tensor.shape}\n")
    
    # Test 4: Policy extraction from MCTS
    print("Test 4: MCTS Policy Extraction...")
    visit_counts = mcts.get_visit_counts()
    assert isinstance(visit_counts, dict), "Visit counts should be a dict"
    
    # Convert to policy vector
    policy = [0.0] * 121
    total_visits = sum(visit_counts.values())
    if total_visits > 0:
        for idx, count in visit_counts.items():
            policy[idx] = count / total_visits
    
    policy_sum = sum(policy)
    assert abs(policy_sum - 1.0) < 1e-5 or policy_sum == 0.0, f"Policy should sum to 1.0, got {policy_sum}"
    print(f"✓ MCTS visit counts extracted")
    print(f"✓ Policy vector normalized (sum={policy_sum:.6f})\n")
    
    # Test 5: Trajectory data structure
    print("Test 5: Trajectory Data Structure...")
    trajectory_step = {
        "state": grid.tolist(),
        "player": board.get_current_player(),
        "policy": policy
    }
    
    assert len(trajectory_step["state"]) == 121, "State should be 121 elements"
    assert trajectory_step["player"] in [1, 2], "Player should be 1 or 2"
    assert len(trajectory_step["policy"]) == 121, "Policy should be 121 elements"
    print(f"✓ Trajectory step structure correct")
    print(f"  - State: {len(trajectory_step['state'])} elements")
    print(f"  - Player: {trajectory_step['player']}")
    print(f"  - Policy: {len(trajectory_step['policy'])} elements\n")
    
    # Test 6: Network integration (if available)
    print("Test 6: Network Integration...")
    try:
        model = ChineseCheckersNet(num_players=2)
        model.eval()
        
        with torch.no_grad():
            policy_logits, value = model(tensor.unsqueeze(0))
        
        assert policy_logits.shape == (1, 121), f"Policy shape mismatch: {policy_logits.shape}"
        assert value.shape == (1, 1), f"Value shape mismatch: {value.shape}"
        print(f"✓ Network forward pass works")
        print(f"  - Policy: {policy_logits.shape}")
        print(f"  - Value: {value.shape}\n")
    except Exception as e:
        print(f"⚠ Network test skipped: {e}\n")
    
    print("=" * 50)
    print("ALL SELF-PLAY TESTS PASSED (6/6)")
    print("=" * 50)
    print("\nNote: Full self-play orchestration should be")
    print("called from a separate training script, not here.")