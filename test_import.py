#!/usr/bin/env python3
"""Minimal test to diagnose C++ module import issue"""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    sys.path.insert(0, 'python')
    print("Attempting to import cc_core...")
    import cc_core
    print("✓ Module imported successfully!")
    
    # Try basic operations
    print("\nTesting basic operations...")
    cc_core.MoveGen.initialize()
    print("✓ MoveGen initialized")
    
    board = cc_core.Board(2)
    print("✓ Board created")
    
    grid = board.get_grid()
    print(f"✓ Grid retrieved: shape={grid.shape}, size={len(grid)}")
    
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
