#include "board.hpp"
#include <algorithm>

// Define the 10-pin triangle indices (Simplified example indices)
// You will replace these with the actual 0-120 indices of your 13x13 map
const std::map<int, std::vector<int>> TARGET_TRIANGLES = {
    {1, {111, 112, 113, 114, 115, 116, 117, 118, 119, 120}}, // South (Target for North)
    {2, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}}                       // North (Target for South)
};

Board::Board(int num_players) : num_players(num_players), current_player(1) {
    grid.fill(0);
    
    // Initialize marbles in Home Triangles
    for (int p = 1; p <= num_players; ++p) {
        // In a real implementation, p=1 starts in North, p=2 starts in South
        auto home = (p == 1) ? TARGET_TRIANGLES.at(2) : TARGET_TRIANGLES.at(1);
        for (int idx : home) {
            grid[idx] = (int8_t)p;
        }
    }
}

void Board::apply_move(const Move& m) {
    grid[m.to] = grid[m.from];
    grid[m.from] = 0;
    next_turn();
}

void Board::undo_move(const Move& m, int8_t original_player_at_to) {
    // Note: undoing requires knowing who is moving back. 
    // Usually, it's the player whose turn it WAS.
    int prev_player = (current_player == 1) ? num_players : current_player - 1;
    grid[m.from] = (int8_t)prev_player;
    grid[m.to] = original_player_at_to;
    current_player = prev_player;
}

void Board::next_turn() {
    current_player = (current_player % num_players) + 1;
}

bool Board::check_player_won(int player_id) const {
    const auto& target = TARGET_TRIANGLES.at(player_id);
    for (int idx : target) {
        if (grid[idx] != player_id) return false;
    }
    return true;
}

bool Board::is_terminal() const {
    for (int p = 1; p <= num_players; ++p) {
        if (check_player_won(p)) return true;
    }
    return false;
}

RewardVector Board::get_rewards() const {
    RewardVector rewards(num_players, -1.0f); // Default to loss
    for (int p = 1; p <= num_players; ++p) {
        if (check_player_won(p)) {
            rewards[p - 1] = 1.0f; // Found the winner
            return rewards;
        }
    }
    return rewards; // Should only be called if terminal
}