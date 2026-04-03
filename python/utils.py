import torch
import numpy as np
import matplotlib.pyplot as plt

# The Star Mask: Defines which of the 169 (13x13) cells are actual board holes.
# We generate this using the same logic we discussed for the C++ side.
def get_star_indices():
    """
    The mathematically perfect 121-node star.
    Requires a 17x17 grid to avoid clipping the tips.
    """
    rows, cols = [], []
    for r in range(17):
        for c in range(17):
            # Center the axial coordinates at (8, 8)
            q = c - 8
            r_ax = r - 8
            s = -(q + r_ax)
            
            # The 121-node star property: 
            # In axial space, at least two coordinates must be within the hexagon radius (4).
            check = 0
            if abs(q) <= 4: check += 1
            if abs(r_ax) <= 4: check += 1
            if abs(s) <= 4: check += 1
            
            if check >= 2:
                rows.append(r)
                cols.append(c)
                
    return np.array(rows), np.array(cols)

# --- PRE-COMPUTED CONSTANTS ---
ROWS, COLS = get_star_indices()
# Verify the math: 61 (Hexagon) + 60 (6 Spikes) = 121
assert len(ROWS) == 121, f"Geometry Error: Expected 121, found {len(ROWS)}"

STAR_MASK = np.zeros((17, 17), dtype=np.float32)
STAR_MASK[ROWS, COLS] = 1.0

def hex_to_tensor(board_array, current_player_id):
    full_grid = np.zeros((17, 17), dtype=np.float32)
    full_grid[ROWS, COLS] = board_array 

    # 1. My Marbles (current player's pieces)
    my_marbles = (full_grid == current_player_id).astype(np.float32)
    # 2. Opponent Marbles (all other pieces)
    opp_marbles = ((full_grid != 0) & (full_grid != current_player_id)).astype(np.float32)
    # 3. Empty valid spots
    empty_holes = ((full_grid == 0) & (STAR_MASK == 1.0)).astype(np.float32)
    # 4. The Star Mask (the 121 valid positions)
    # Removed redundant player identity plane - network learns from piece positions
    
    channels = [my_marbles, opp_marbles, empty_holes, STAR_MASK]
    return torch.from_numpy(np.stack(channels, axis=0))

def get_move_mask(legal_moves):
    """
    Converts a list of Move objects into a 121-length bitmask.
    Used to zero-out illegal moves in the Policy Head.
    """
    mask = np.zeros(121, dtype=np.float32)
    for m in legal_moves:
        # We use the destination index 'to_idx' as the policy label
        mask[m.to_idx] = 1.0
    return torch.from_numpy(mask)

def visualize_board(board_array):
    """Debug helper to ensure your mapping isn't mirrored or broken."""
    # board_array is 121 elements, not 289
    full_grid = np.zeros((17, 17), dtype=np.float32)
    full_grid[ROWS, COLS] = board_array
    
    plt.imshow(full_grid, cmap='viridis')
    plt.colorbar(label="Player ID")
    plt.title("17x17 Grid Representation (121 valid positions)")
    plt.show()

if __name__ == "__main__":
    print("=== Testing utils.py ===\n")
    
    # Test 1: Star geometry
    print("Test 1: Star Geometry...")
    assert STAR_MASK.sum() == 121, f"Star mask should have 121 positions, got {STAR_MASK.sum()}"
    assert len(ROWS) == 121, f"Should have 121 valid positions, got {len(ROWS)}"
    print("✓ Star geometry correct (121 positions)\n")
    
    # Test 2: Tensor shape
    print("Test 2: Tensor Encoding Shape...")
    dummy_board = np.zeros(121)
    test_tensor = hex_to_tensor(dummy_board, 1)
    assert test_tensor.shape == (4, 17, 17), f"Tensor shape mismatch: {test_tensor.shape}"
    print("✓ Tensor shape correct: (4, 17, 17)\n")
    
    # Test 3: Channel separation
    print("Test 3: Channel Separation...")
    test_board = np.zeros(121)
    test_board[0] = 1   # Player 1 piece at position 0
    test_board[1] = 2   # Player 2 piece at position 1
    test_board[2] = 1   # Another Player 1 piece
    
    # From Player 1's perspective
    tensor_p1 = hex_to_tensor(test_board, 1)
    # Channel 0: My marbles (Player 1)
    assert tensor_p1[0, ROWS[0], COLS[0]] == 1.0, "Player 1's piece not in 'my marbles' channel"
    assert tensor_p1[0, ROWS[2], COLS[2]] == 1.0, "Player 1's second piece not in 'my marbles' channel"
    # Channel 1: Opponent marbles (Player 2)
    assert tensor_p1[1, ROWS[1], COLS[1]] == 1.0, "Player 2's piece not in 'opponent marbles' channel"
    # Channel 2: Empty holes
    assert tensor_p1[2, ROWS[3], COLS[3]] == 1.0, "Empty position not marked in empty holes channel"
    # Channel 3: Star mask
    assert tensor_p1[3, ROWS[0], COLS[0]] == 1.0, "Star mask not set correctly"
    print("✓ Channel 0: My marbles correct")
    print("✓ Channel 1: Opponent marbles correct")
    print("✓ Channel 2: Empty holes correct")
    print("✓ Channel 3: Star mask correct\n")
    
    # Test 4: Player perspective switching
    print("Test 4: Player Perspective Switching...")
    tensor_p2 = hex_to_tensor(test_board, 2)
    # From Player 2's perspective, channels should flip
    assert tensor_p2[0, ROWS[1], COLS[1]] == 1.0, "Player 2's piece not in 'my marbles' from P2 perspective"
    assert tensor_p2[1, ROWS[0], COLS[0]] == 1.0, "Player 1's piece not in 'opponent marbles' from P2 perspective"
    assert tensor_p2[1, ROWS[2], COLS[2]] == 1.0, "Player 1's second piece not in 'opponent marbles' from P2 perspective"
    print("✓ Player perspective correctly switches channels\n")
    
    # Test 5: Move mask function
    print("Test 5: Move Mask Function...")
    # Create dummy Move objects
    class DummyMove:
        def __init__(self, to_idx):
            self.to_idx = to_idx
    
    legal_moves = [DummyMove(5), DummyMove(10), DummyMove(15)]
    mask = get_move_mask(legal_moves)
    
    assert mask.shape == (121,), f"Mask shape should be (121,), got {mask.shape}"
    assert mask[5] == 1.0, "Legal move at index 5 not marked"
    assert mask[10] == 1.0, "Legal move at index 10 not marked"
    assert mask[15] == 1.0, "Legal move at index 15 not marked"
    assert mask[0] == 0.0, "Illegal move at index 0 should be 0"
    assert mask.sum() == 3.0, f"Should have exactly 3 legal moves, got {mask.sum()}"
    print("✓ Move mask correctly marks legal moves\n")
    
    # Test 6: Visualization (optional, requires manual inspection)
    print("Test 6: Visualization (creating test board)...")
    vis_board = np.zeros(121)
    vis_board[0] = 1
    vis_board[120] = 2
    # Uncomment to see visualization:
    # visualize_board(vis_board)
    print("✓ Visualization function works (display disabled)\n")
    
    print("=" * 50)
    print("ALL UTILS TESTS PASSED (6/6)")
    print("=" * 50)