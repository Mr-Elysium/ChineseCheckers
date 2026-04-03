#ifndef TYPES_HPP
#define TYPES_HPP

#include <array>
#include <vector>
#include <stdint.h>

// The board has 121 positions (compact sequential indexing).
// 0 = empty, 1-6 = Player IDs.
using BoardArray = std::array<int8_t, 121>;

// RewardVector[i] is the win probability for Player (i+1).
// For 2 players, size is 2. For 6 players, size is 6.
using RewardVector = std::vector<float>;

struct Move {
    int from_idx;
    int to_idx;

    // Helpful for pybind11 and debugging
    bool operator==(const Move& other) const {
        return from_idx == other.from_idx && to_idx == other.to_idx;
    }

    // Add this explicit constructor
    Move(int f, int t) : from_idx(f), to_idx(t) {}

    // Also add a default constructor (good practice for MCTS)
    Move() : from_idx(-1), to_idx(-1) {}
};

#endif