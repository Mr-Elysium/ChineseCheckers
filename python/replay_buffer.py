import numpy as np
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, max_size=50000):
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_trajectory(self, trajectory):
        for step in trajectory:
            self.buffer.append(step)
    
    def sample_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([step["state"] for step in batch], dtype=np.float32)
        policies = np.array([step["policy"] for step in batch], dtype=np.float32)
        rewards = np.array([step["rewards"] for step in batch], dtype=np.float32)
        players = np.array([step["player"] for step in batch], dtype=np.int32)
        
        return states, policies, rewards, players
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        self.buffer.clear()

if __name__ == "__main__":
    print("=== Testing replay_buffer.py ===\n")
    
    # Test 1: Buffer initialization
    print("Test 1: Buffer Initialization...")
    buffer = ReplayBuffer(max_size=1000)
    assert len(buffer) == 0, "Buffer should start empty"
    print("✓ Buffer initialized correctly\n")
    
    # Test 2: Add trajectory
    print("Test 2: Add Trajectory...")
    dummy_trajectory = [
        {
            "state": np.zeros(121),
            "player": 1,
            "policy": np.ones(121) / 121,
            "rewards": [1.0, -1.0]
        },
        {
            "state": np.ones(121),
            "player": 2,
            "policy": np.ones(121) / 121,
            "rewards": [1.0, -1.0]
        }
    ]
    
    buffer.add_trajectory(dummy_trajectory)
    assert len(buffer) == 2, f"Buffer should have 2 steps, got {len(buffer)}"
    print("✓ Trajectory added correctly\n")
    
    # Test 3: Sample batch
    print("Test 3: Sample Batch...")
    states, policies, rewards, players = buffer.sample_batch(2)
    
    assert states.shape == (2, 121), f"States shape mismatch: {states.shape}"
    assert policies.shape == (2, 121), f"Policies shape mismatch: {policies.shape}"
    assert rewards.shape == (2, 2), f"Rewards shape mismatch: {rewards.shape}"
    assert players.shape == (2,), f"Players shape mismatch: {players.shape}"
    print("✓ Batch sampling works correctly\n")
    
    # Test 4: Max size enforcement
    print("Test 4: Max Size Enforcement...")
    small_buffer = ReplayBuffer(max_size=5)
    for i in range(10):
        small_buffer.add_trajectory([{
            "state": np.zeros(121),
            "player": 1,
            "policy": np.ones(121) / 121,
            "rewards": [0.0, 0.0]
        }])
    
    assert len(small_buffer) == 5, f"Buffer should be capped at 5, got {len(small_buffer)}"
    print("✓ Max size enforced correctly\n")
    
    # Test 5: Clear buffer
    print("Test 5: Clear Buffer...")
    buffer.clear()
    assert len(buffer) == 0, "Buffer should be empty after clear"
    print("✓ Buffer cleared correctly\n")
    
    print("=" * 50)
    print("ALL REPLAY BUFFER TESTS PASSED (5/5)")
    print("=" * 50)
