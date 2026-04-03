# Chinese Checkers UI

Play against your trained AlphaZero model with either a CLI or beautiful Pygame GUI!

## Quick Start

### CLI Mode (Terminal)
```bash
python ui/play.py
```

### GUI Mode (Pygame - Recommended)
```bash
# Install pygame first
pip install pygame

# Run with GUI
python ui/play.py --gui
```

## Usage

### Basic Commands

**Play with GUI:**
```bash
python ui/play.py --gui
```

**Play in terminal:**
```bash
python ui/play.py
```

**Use a specific model:**
```bash
python ui/play.py --gui --model checkpoints/best_model.pt
```

**Adjust AI difficulty:**
```bash
# Easier (fewer MCTS iterations)
python ui/play.py --gui --mcts-iterations 50

# Harder (more MCTS iterations)
python ui/play.py --gui --mcts-iterations 500
```

**Play as Player 2 (CLI only):**
```bash
python ui/play.py --human-player 2
```

## Features

### CLI Mode (`play_cli.py`)
- ✅ Works everywhere (no dependencies beyond PyTorch)
- ✅ ASCII board visualization with row/col indices
- ✅ Text-based move input
- ✅ Legal move display helper
- ✅ Choose which player you want to be

**Controls:**
- Enter moves as: `from_row from_col to_row to_col` (e.g., `0 8 1 8`)
- Type `moves` to see all legal moves
- Type `quit` to exit

### GUI Mode (`play_gui.py`)
- ✅ Beautiful hexagonal board visualization
- ✅ Mouse-based controls (click to select, click to move)
- ✅ Visual highlighting of selected pieces and legal moves
- ✅ Real-time hover effects
- ✅ Info panel with game state
- ✅ Color-coded players (Blue = You, Red = AI)

**Controls:**
- Click on your piece to select it
- Legal moves will be highlighted in green
- Click on a highlighted position to move
- Click elsewhere to deselect

## File Structure

```
ui/
├── __init__.py       # Package marker
├── play.py           # Main entry point (chooses CLI or GUI)
├── play_cli.py       # Terminal interface
├── play_gui.py       # Pygame GUI interface
└── README.md         # This file
```

## How It Works

### Coordinate System
The game uses a 121-node hexagonal star board:
- **Internal**: 0-120 indices (C++ engine)
- **Visual**: 17×17 grid (what you see)
- **Conversion**: Handled automatically via `ROWS` and `COLS` from `utils.py`

### AI Integration
1. Loads your trained model from checkpoint
2. Runs MCTS search with neural network guidance
3. Selects best move based on visit counts
4. Adjustable difficulty via `--mcts-iterations`

### Game Flow
```
Human Turn → Select Piece → Show Legal Moves → Execute Move
    ↓
AI Turn → MCTS Search → Neural Network Evaluation → Execute Move
    ↓
Check Win Condition → Repeat or Game Over
```

## Tips

### For Best Experience
- Use GUI mode for visual gameplay
- Start with 200 MCTS iterations (balanced)
- Train your model first for better AI opponent
- Use `moves` command in CLI to learn the board

### Difficulty Levels
- **Easy**: `--mcts-iterations 50`
- **Medium**: `--mcts-iterations 200` (default)
- **Hard**: `--mcts-iterations 500`
- **Expert**: `--mcts-iterations 1000+`

### If No Model Available
The game will still work with random AI policy - useful for:
- Testing the interface
- Learning the game rules
- Playing against a weak opponent

## Troubleshooting

**"pygame not installed"**
```bash
pip install pygame
# or
uv pip install pygame
```

**"Model file not found"**
- Train a model first: `python scripts/train.py`
- Or specify a different path: `--model path/to/model.pt`
- Or continue with random AI (will prompt)

**Game runs slowly**
- Reduce MCTS iterations: `--mcts-iterations 100`
- Use CPU mode (default) instead of CUDA for small models

## Examples

### Quick test game (easy AI)
```bash
python ui/play.py --gui --mcts-iterations 50
```

### Serious game (strong AI)
```bash
python ui/play.py --gui --model checkpoints/best_model.pt --mcts-iterations 400
```

### Terminal play (no GUI)
```bash
python ui/play.py --human-player 1
```

Enjoy playing against your AlphaZero AI! 🎮
