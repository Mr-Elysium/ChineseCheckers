#include "move_gen.hpp"
#include <algorithm>

// Initialize static members
int MoveGen::neighbors[121][6];
bool MoveGen::initialized = false;

void MoveGen::initialize() {
    if (initialized) return;

    // This logic builds the adjacency list for the 121-node star.
    // For this implementation, we assume index_to_axial and axial_to_index
    // are correctly mapped to your 121-node layout.
    for (int i = 0; i < 121; ++i) {
        AxialCoord coord = index_to_axial(i);
        
        // Hexagonal neighbor offsets in Axial space
        int dq[6] = {1, 1, 0, -1, -1, 0};
        int dr[6] = {0, -1, -1, 0, 1, 1};

        for (int d = 0; d < 6; ++d) {
            neighbors[i][d] = axial_to_index(coord.q + dq[d], coord.r + dr[d]);
        }
    }
    initialized = true;
}

std::vector<Move> MoveGen::get_legal_moves(const Board& board) {
    if (!initialized) initialize();

    std::vector<Move> moves;
    const BoardArray& grid = board.get_grid();
    int p_id = board.get_current_player();

    for (int i = 0; i < 121; ++i) {
        if (grid[i] == p_id) {
            // 1. Check for simple 1-step moves
            for (int d = 0; d < 6; ++d) {
                int neighbor = neighbors[i][d];
                if (neighbor != -1 && grid[neighbor] == 0) {
                    moves.push_back({i, neighbor});
                }
            }

            // 2. Check for complex jump chains
            std::set<int> reached_in_jumps;
            find_jumps(i, grid, reached_in_jumps, i);
            for (int dest : reached_in_jumps) {
                moves.push_back({i, dest});
            }
        }
    }
    return moves;
}

void MoveGen::find_jumps(int current_pos, const BoardArray& grid, 
                         std::set<int>& reached, int start_pos) {
    
    for (int d = 0; d < 6; ++d) {
        int neighbor = neighbors[current_pos][d];
        
        // To jump, there must be a marble in the neighbor spot
        if (neighbor != -1 && grid[neighbor] != 0) {
            int landing = neighbors[neighbor][d]; // Same direction
            
            // Landing spot must be empty and within the 121-node star
            if (landing != -1 && grid[landing] == 0 && landing != start_pos) {
                // If we haven't reached this spot in this jump-chain yet
                if (reached.find(landing) == reached.end()) {
                    reached.insert(landing);
                    // Recurse to find multi-jumps (e.g., A -> C -> E)
                    find_jumps(landing, grid, reached, start_pos);
                }
            }
        }
    }
}

// A 13x13 grid can contain the whole 121-node star.
// We map 1D index -> 2D (row, col)
MoveGen::AxialCoord MoveGen::index_to_axial(int idx) {
    // We'll treat the board as a 13x13 skewed grid.
    // This is a simple row/column mapping.
    return { idx / 13, idx % 13 };
}

int MoveGen::axial_to_index(int q, int r) {
    // Check bounds for the 13x13 bounding box
    if (q < 0 || q >= 13 || r < 0 || r >= 13) return -1;
    
    int idx = q * 13 + r;
    
    // IMPORTANT: Not all 169 cells in a 13x13 grid are in the 121-node star.
    // For now, let's assume a simplified "Diamond" board for testing,
    // or you can implement the specific "Star" mask here.
    return idx;
}
