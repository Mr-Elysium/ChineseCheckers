#include "move_gen.hpp"
#include <algorithm>
#include <cmath>

// Initialize static members
int MoveGen::neighbors[121][6];
bool MoveGen::initialized = false;

// Build the list of 121 valid star positions in 17x17 grid
static void build_star_positions(std::vector<std::pair<int, int>>& positions) {
    positions.clear();
    for (int r = 0; r < 17; ++r) {
        for (int c = 0; c < 17; ++c) {
            // Check if (r, c) is a valid star position
            int q = c - 8;
            int r_ax = r - 8;
            int s = -(q + r_ax);
            
            int check = 0;
            if (std::abs(q) <= 4) check++;
            if (std::abs(r_ax) <= 4) check++;
            if (std::abs(s) <= 4) check++;
            
            if (check >= 2) {
                positions.push_back({r, c});
            }
        }
    }
}

// Get star positions (lazy initialization)
static const std::vector<std::pair<int, int>>& get_star_positions() {
    static std::vector<std::pair<int, int>> positions;
    static bool built = false;
    if (!built) {
        build_star_positions(positions);
        built = true;
    }
    return positions;
}

void MoveGen::initialize() {
    if (initialized) return;

    const auto& star_positions = get_star_positions();
    
    // Build adjacency list for 121 positions
    for (int i = 0; i < 121; ++i) {
        int r = star_positions[i].first;
        int c = star_positions[i].second;
        
        // Hexagonal neighbor offsets (6 directions)
        int dr[6] = {-1, -1, 0, 1, 1, 0};
        int dc[6] = {0, 1, 1, 0, -1, -1};

        for (int d = 0; d < 6; ++d) {
            int nr = r + dr[d];
            int nc = c + dc[d];
            
            // Find if (nr, nc) is in our 121 positions
            neighbors[i][d] = -1;
            for (int j = 0; j < 121; ++j) {
                if (star_positions[j].first == nr && star_positions[j].second == nc) {
                    neighbors[i][d] = j;
                    break;
                }
            }
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

// Helper functions for coordinate conversion (kept for compatibility)
MoveGen::AxialCoord MoveGen::index_to_axial(int idx) {
    if (idx < 0 || idx >= 121) return {-1, -1};
    const auto& star_positions = get_star_positions();
    int r = star_positions[idx].first;
    int c = star_positions[idx].second;
    return {r, c};
}

int MoveGen::axial_to_index(int q, int r) {
    const auto& star_positions = get_star_positions();
    // Find the index of position (q, r) in our 121 positions
    for (int i = 0; i < 121; ++i) {
        if (star_positions[i].first == q && star_positions[i].second == r) {
            return i;
        }
    }
    return -1;
}
