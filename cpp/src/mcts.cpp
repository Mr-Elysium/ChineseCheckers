#include "mcts.hpp"
#include "move_gen.hpp"
#include <cmath>
#include <algorithm>

void MCTS::search(int iterations, Board& board, Predictor predictor) {
    // 1. Initialize root if it doesn't exist
    if (!root) {
        auto grid = board.get_grid();
        auto [policy, value] = predictor(std::vector<int>(grid.begin(), grid.end()));
        root = std::make_unique<MCTSNode>(Move(-1, -1), 1.0f);
        expand_node(root.get(), board, policy);
    }

    for (int i = 0; i < iterations; ++i) {
        MCTSNode* curr = root.get();
        Board temp_board = board;
        std::vector<MCTSNode*> path = {curr};

        // PHASE 1: SELECTION
        // Move down the tree using PUCT until we hit a leaf
        while (!curr->children.empty()) {
            curr = select_child(curr);
            temp_board.apply_move(curr->move);
            path.push_back(curr);
        }

        // PHASE 2 & 3: EXPANSION & EVALUATION
        float leaf_value = 0.0f;
        if (temp_board.is_terminal()) {
            // Use the root player's perspective for terminal rewards
            leaf_value = temp_board.get_rewards()[board.get_current_player() - 1];
        } else {
            auto grid = temp_board.get_grid();
            auto [policy, value] = predictor(std::vector<int>(grid.begin(), grid.end()));
            
            expand_node(curr, temp_board, policy);
            // CRITICAL FIX: Evaluate from the perspective of the player at the leaf position
            // The NN returns value from current player's perspective at temp_board
            // We need to convert this to root player's perspective
            int leaf_player = temp_board.get_current_player();
            int root_player = board.get_current_player();
            
            if (leaf_player == root_player) {
                // Same player, use value directly
                leaf_value = value[root_player - 1];
            } else {
                // Different player in 2-player game, negate the value
                // In 2-player zero-sum: opponent's win = our loss
                leaf_value = -value[leaf_player - 1];
            }
        }

        // PHASE 4: BACKPROPAGATION
        for (auto node : path) {
            node->visit_count++;
            node->value_sum += leaf_value;
        }
    }
}

MCTSNode* MCTS::select_child(MCTSNode* node) {
    MCTSNode* best_child = nullptr;
    float best_score = -1e9f;
    
    int total_visits = 0;
    for (auto const& [idx, child] : node->children) {
        total_visits += child->visit_count;
    }

    for (auto const& [idx, child] : node->children) {
        // PUCT Formula: Q + U
        // U = C_puct * P(s,a) * sqrt(Sum_N) / (1 + n)
        float q = child->get_value();
        float u = cpuct * child->prior * std::sqrt((float)total_visits) / (1.0f + child->visit_count);
        float score = q + u;

        if (score > best_score) {
            best_score = score;
            best_child = child.get();
        }
    }
    return best_child;
}

void MCTS::expand_node(MCTSNode* node, Board& board, const std::vector<float>& policy) {
    auto legal_moves = MoveGen::get_legal_moves(board);
    for (auto const& m : legal_moves) {
        // We use the target index 'to_idx' as the policy index
        float prior = (m.to_idx >= 0 && m.to_idx < (int)policy.size()) ? policy[m.to_idx] : 0.0f;
        node->children[m.to_idx] = std::make_unique<MCTSNode>(m, prior);
    }
}

Move MCTS::get_best_move() {
    if (!root || root->children.empty()) return Move(-1, -1);

    MCTSNode* best_child = nullptr;
    int max_visits = -1;

    for (auto const& [idx, child] : root->children) {
        if (child->visit_count > max_visits) {
            max_visits = child->visit_count;
            best_child = child.get();
        }
    }
    
    Move m = best_child->move;
    // Reset the root for the next move to prevent memory explosion
    root.reset(); 
    return m;
}

Move MCTS::get_best_move_and_reuse() {
    if (!root || root->children.empty()) return Move(-1, -1);

    MCTSNode* best_child = nullptr;
    int max_visits = -1;
    int best_idx = -1;

    for (auto const& [idx, child] : root->children) {
        if (child->visit_count > max_visits) {
            max_visits = child->visit_count;
            best_child = child.get();
            best_idx = idx;
        }
    }
    
    Move m = best_child->move;
    
    // Reuse the subtree: move the best child to become the new root
    if (best_idx != -1 && root->children.find(best_idx) != root->children.end()) {
        root = std::move(root->children[best_idx]);
    } else {
        root.reset();
    }
    
    return m;
}

void MCTS::clear_tree() {
    root.reset();
}

std::map<int, int> MCTS::get_visit_counts() {
    std::map<int, int> counts;
    if (root) {
        for (auto const& [idx, child] : root->children) {
            counts[idx] = child->visit_count;
        }
    }
    return counts;
}