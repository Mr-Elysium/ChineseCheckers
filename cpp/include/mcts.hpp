#ifndef MCTS_HPP
#define MCTS_HPP

#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "board.hpp"
#include "types.hpp"

// The Python callback: takes board state (vector<int>), returns (policy_vector, value_vector)
using Predictor = std::function<std::pair<std::vector<float>, std::vector<float>>(std::vector<int>)>;

struct MCTSNode {
    Move move;
    float prior;
    int visit_count = 0;
    float value_sum = 0.0f;
    std::map<int, std::unique_ptr<MCTSNode>> children;

    MCTSNode(Move m, float p) : move(m), prior(p) {}

    float get_value() const {
        return visit_count == 0 ? 0.0f : value_sum / visit_count;
    }
};

class MCTS {
public:
    explicit MCTS(float cpuct = 1.41f) : cpuct(cpuct) {}
    
    // Core AlphaZero search
    void search(int iterations, Board& board, Predictor predictor);
    
    // Selection logic
    Move get_best_move();
    
    // Move selection with tree reuse
    Move get_best_move_and_reuse();
    
    // Returns visit counts for training labels
    std::map<int, int> get_visit_counts();
    
    // Clear the tree (for new games)
    void clear_tree();

private:
    float cpuct;
    std::unique_ptr<MCTSNode> root;

    MCTSNode* select_child(MCTSNode* node);
    void expand_node(MCTSNode* node, Board& board, const std::vector<float>& policy);
};

#endif