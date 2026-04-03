#ifndef MOVE_GEN_HPP
#define MOVE_GEN_HPP

#include "types.hpp"
#include "board.hpp"
#include <vector>
#include <set>

class MoveGen {
public:
    // Builds the adjacency map for the 121-node star
    static void initialize();

    // The primary entry point for MCTS
    static std::vector<Move> get_legal_moves(const Board& board);

private:
    // Helper for recursive multi-hop jumps
    static void find_jumps(int current_pos, 
                           const BoardArray& grid, 
                           std::set<int>& reached, 
                           int start_pos);

    // Static lookup: neighbors[121][6]. -1 if no neighbor in that direction.
    static int neighbors[121][6];
    static bool initialized;

    // Internal geometry helpers used only during initialize()
    struct AxialCoord { int q, r; };
    static AxialCoord index_to_axial(int idx);
    static int axial_to_index(int q, int r);
};

#endif