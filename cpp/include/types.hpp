#ifndef TYPES_HPP
#define TYPES_HPP

#include <array>
#include <vector>
#include <stdint.h>

// The board has 121 holes. 0 = empty, 1-6 = Player IDs.
using BoardArray = std::array<int8_t, 121>;

// RewardVector[i] is the win probability for Player (i+1).
// For 2 players, size is 2. For 6 players, size is 6.
using RewardVector = std::vector<float>;

struct Move {
    int from;
    int to;

    // Helpful for pybind11 and debugging
    bool operator==(const Move& other) const {
        return from == other.from && to == other.to;
    }
};

#endif