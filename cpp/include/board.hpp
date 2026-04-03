#ifndef BOARD_HPP
#define BOARD_HPP

#include "types.hpp"
#include <vector>
#include <map>

class Board {
public:
    Board(int num_players = 2);

    // State management
    void apply_move(const Move& m);
    void undo_move(const Move& m, int8_t original_player_at_to); // 0 usually
    
    // Getters
    const BoardArray& get_grid() const { return grid; }
    int get_current_player() const { return current_player; }
    int get_num_players() const { return num_players; }

    // Win Logic
    bool is_terminal() const;
    RewardVector get_rewards() const; // Returns 1.0 for winner, -1.0 for others

    // Static Helpers for geometry
    static const std::vector<int>& get_target_triangle(int player_id);
    static const std::vector<int>& get_home_triangle(int player_id);

private:
    BoardArray grid;
    int current_player;
    int num_players;

    void next_turn();
    bool check_player_won(int player_id) const;
};

#endif