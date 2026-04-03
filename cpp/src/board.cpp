#include "board.hpp"
#include <algorithm>

// Triangle definitions using sequential 0-120 indices
// The 121 positions are ordered by scanning the 17x17 grid row-by-row,
// collecting only valid star positions

// North triangle: first 10 positions (indices 0-9)
const std::vector<int> NORTH_TRIANGLE = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

// South triangle: last 10 positions (indices 111-120)
const std::vector<int> SOUTH_TRIANGLE = {111, 112, 113, 114, 115, 116, 117, 118, 119, 120};

// Map player IDs to their home and target triangles
const std::map<int, std::vector<int>> HOME_TRIANGLES = {
    {1, NORTH_TRIANGLE},  // Player 1 starts in North
    {2, SOUTH_TRIANGLE}   // Player 2 starts in South
};

const std::map<int, std::vector<int>> TARGET_TRIANGLES = {
    {1, SOUTH_TRIANGLE},  // Player 1 targets South
    {2, NORTH_TRIANGLE}   // Player 2 targets North
};

Board::Board(int num_players) : num_players(num_players), current_player(1) {
    // Initialize all positions as empty
    grid.fill(0);
    
    // Place marbles in home triangles
    for (int p = 1; p <= num_players; ++p) {
        const auto& home = HOME_TRIANGLES.at(p);
        for (int idx : home) {
            grid[idx] = (int8_t)p;
        }
    }
}

void Board::apply_move(const Move& m) {
    grid[m.to_idx] = grid[m.from_idx];
    grid[m.from_idx] = 0;
    next_turn();
}

void Board::undo_move(const Move& m, int8_t original_player_at_to) {
    // Note: undoing requires knowing who is moving back. 
    // Usually, it's the player whose turn it WAS.
    int prev_player = (current_player == 1) ? num_players : current_player - 1;
    grid[m.from_idx] = (int8_t)prev_player;
    grid[m.to_idx] = original_player_at_to;
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

const std::vector<int>& Board::get_target_triangle(int player_id) {
    return TARGET_TRIANGLES.at(player_id);
}

const std::vector<int>& Board::get_home_triangle(int player_id) {
    return HOME_TRIANGLES.at(player_id);
}