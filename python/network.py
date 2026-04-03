import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    """
    Standard ResNet block: Two Conv layers with Batch Normalization 
    and a skip connection.
    """
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class ChineseCheckersNet(nn.Module):
    def __init__(self, in_channels=4, num_res_blocks=10, num_channels=128, num_players=2):
        super(ChineseCheckersNet, self).__init__()
        self.num_players = num_players
        
        # 1. Initial Convolutional Block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # 2. Residual Backbone
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # 3. Policy Head (Which move to make?)
        # We output 121 values corresponding to the target hole indices.
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 17 * 17, 121) 
        )

        # 4. Value Head (Who is winning?)
        # For 2-player: outputs single value in [-1, 1] (current player's win probability)
        # For 6-player: outputs probability distribution over players (softmax)
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 17 * 17, 128)
        self.value_fc2 = nn.Linear(128, num_players if num_players > 2 else 1)

    def forward(self, x):
        """
        Input x: Tensor of shape (Batch, Channels, 17, 17)
        Returns: 
            policy_logits: (Batch, 121)
            value: (Batch, 1) for 2-player or (Batch, num_players) for 6-player
        """
        out = self.conv_block(x)
        for block in self.res_blocks:
            out = block(out)
        
        policy = self.policy_head(out)
        
        # Value head with proper activation based on num_players
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        value = self.value_fc2(v)
        
        if self.num_players == 2:
            # 2-player: tanh for [-1, 1] range
            value = torch.tanh(value)
        else:
            # 6-player: softmax for probability distribution
            value = F.softmax(value, dim=-1)
        
        return policy, value

# --- Shape Verification ---
if __name__ == "__main__":
    print("=== Testing network.py ===\n")
    
    # Test 1: 2-player network shapes
    print("Test 1: 2-Player Network Shapes...")
    model_2p = ChineseCheckersNet(in_channels=4, num_res_blocks=10, num_players=2)
    dummy_input = torch.randn(1, 4, 17, 17)
    
    p, v = model_2p(dummy_input)
    
    assert p.shape == (1, 121), f"Policy shape mismatch: {p.shape}"
    assert v.shape == (1, 1), f"Value shape mismatch: {v.shape}"
    print(f"✓ Policy shape: {p.shape}")
    print(f"✓ Value shape:  {v.shape}\n")
    
    # Test 2: Value range for 2-player (should be in [-1, 1])
    print("Test 2: 2-Player Value Range...")
    assert v.item() >= -1.0 and v.item() <= 1.0, f"Value {v.item()} outside [-1, 1] range"
    print(f"✓ Value in valid range: {v.item():.4f}\n")
    
    # Test 3: 6-player network shapes
    print("Test 3: 6-Player Network Shapes...")
    model_6p = ChineseCheckersNet(in_channels=4, num_res_blocks=10, num_players=6)
    p6, v6 = model_6p(dummy_input)
    
    assert p6.shape == (1, 121), f"Policy shape mismatch: {p6.shape}"
    assert v6.shape == (1, 6), f"Value shape mismatch: {v6.shape}"
    print(f"✓ Policy shape: {p6.shape}")
    print(f"✓ Value shape:  {v6.shape}\n")
    
    # Test 4: Softmax sum for 6-player (should sum to 1.0)
    print("Test 4: 6-Player Value Softmax...")
    value_sum = v6.sum().item()
    assert abs(value_sum - 1.0) < 1e-5, f"Softmax should sum to 1.0, got {value_sum}"
    print(f"✓ Softmax sums to 1.0: {value_sum:.6f}\n")
    
    # Test 5: Batch processing
    print("Test 5: Batch Processing...")
    batch_input = torch.randn(8, 4, 17, 17)
    p_batch, v_batch = model_2p(batch_input)
    
    assert p_batch.shape == (8, 121), f"Batch policy shape mismatch: {p_batch.shape}"
    assert v_batch.shape == (8, 1), f"Batch value shape mismatch: {v_batch.shape}"
    print(f"✓ Batch size 8 works correctly")
    print(f"  Policy: {p_batch.shape}")
    print(f"  Value:  {v_batch.shape}\n")
    
    # Test 6: Model parameters
    print("Test 6: Model Parameters...")
    total_params = sum(p.numel() for p in model_2p.parameters())
    model_size_mb = total_params * 4 / 1024 / 1024
    
    assert total_params > 100000, "Model seems too small"
    assert total_params < 50000000, "Model seems too large"
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Model size: ~{model_size_mb:.1f} MB (FP32)\n")
    
    # Test 7: Gradient flow (ensure model is trainable)
    print("Test 7: Gradient Flow...")
    model_2p.train()
    optimizer = torch.optim.Adam(model_2p.parameters(), lr=0.001)
    
    # Forward pass
    p_train, v_train = model_2p(dummy_input)
    
    # Dummy loss
    target_policy = torch.randn(1, 121)
    target_value = torch.randn(1, 1)
    loss = F.mse_loss(p_train, target_policy) + F.mse_loss(v_train, target_value)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Check gradients exist
    has_gradients = any(p.grad is not None for p in model_2p.parameters())
    assert has_gradients, "No gradients computed"
    print(f"✓ Gradients computed successfully")
    print(f"✓ Model is trainable\n")
    
    print("=" * 50)
    print("ALL NETWORK TESTS PASSED (7/7)")
    print("=" * 50)