#ifndef MCTS_HPP
#define MCTS_HPP

#include "types.hpp"
#include "board.hpp"
#include "move_gen.hpp"
#include <memory>
#include <map>
#include <mutex>
#include <cmath>

class MCTSNode {
public:
    MCTSNode(MCTSNode* parent, float prior);
    
    // AlphaZero/Max-n statistics
    int visit_count;
    int virtual_loss;
    float prior;
    RewardVector value_sum; // Vector of size N
    
    MCTSNode* parent;
    std::map<int, std::unique_ptr<MCTSNode>> children; // Key is 'to_idx'
    std::mutex node_mutex;

    bool is_expanded() const { return !children.empty(); }
    float get_value(int player_index) const; 
};

class MCTS {
public:
    MCTS(float c_puct = 1.41f);
    
    // Main entry point from Python
    void search(int iterations, Board& root_board, 
                const std::function<std::pair<std::vector<float>, RewardVector>(const BoardArray&)>& predictor);

    Move get_best_move(const Board& root_board);

private:
    float c_puct;
    
    MCTSNode* select(MCTSNode* node, const Board& board);
    void expand(MCTSNode* node, const Board& board, const std::vector<float>& policy_logits);
    void backpropagate(MCTSNode* node, const RewardVector& values);
    
    // Virtual Loss helpers
    void apply_virtual_loss(MCTSNode* node);
    void remove_virtual_loss(MCTSNode* node);
};

#endif