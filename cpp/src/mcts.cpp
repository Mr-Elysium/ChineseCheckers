#include "mcts.hpp"

MCTSNode::MCTSNode(MCTSNode* parent, float prior) 
    : parent(parent), prior(prior), visit_count(0), virtual_loss(0) {}

float MCTSNode::get_value(int player_index) const {
    if (visit_count == 0) return 0;
    return value_sum[player_index] / visit_count;
}

MCTS::MCTS(float c_puct) : c_puct(c_puct) {}

MCTSNode* MCTS::select(MCTSNode* node, const Board& board) {
    int p_idx = board.get_current_player() - 1; // 0-indexed for vector
    
    MCTSNode* best_child = nullptr;
    float best_score = -1e9;

    for (auto& [move_to, child] : node->children) {
        std::lock_guard<std::mutex> lock(child->node_mutex);
        
        // PUCT Formula: Q + U
        // We subtract virtual loss from Q to encourage threads to explore elsewhere
        float Q = child->get_value(p_idx) - (child->virtual_loss * 0.1f);
        float U = c_puct * child->prior * std::sqrt(node->visit_count) / (1 + child->visit_count);
        float score = Q + U;

        if (score > best_score) {
            best_score = score;
            best_child = child.get();
        }
    }
    return best_child;
}

void MCTS::expand(MCTSNode* node, const Board& board, const std::vector<float>& policy_logits) {
    auto moves = MoveGen::get_legal_moves(board);
    for (const auto& m : moves) {
        // policy_logits is 121 long; we use the destination 'to' as the index
        node->children[m.to] = std::make_unique<MCTSNode>(node, policy_logits[m.to]);
    }
    node->value_sum.resize(board.get_num_players(), 0.0f);
}

void MCTS::backpropagate(MCTSNode* node, const RewardVector& values) {
    MCTSNode* curr = node;
    while (curr != nullptr) {
        std::lock_guard<std::mutex> lock(curr->node_mutex);
        curr->visit_count++;
        for (size_t i = 0; i < values.size(); ++i) {
            curr->value_sum[i] += values[i];
        }
        curr = curr->parent;
    }
}

void MCTS::apply_virtual_loss(MCTSNode* node) {
    MCTSNode* curr = node;
    while (curr != nullptr) {
        std::lock_guard<std::mutex> lock(curr->node_mutex);
        curr->virtual_loss++;
        curr = curr->parent;
    }
}

void MCTS::remove_virtual_loss(MCTSNode* node) {
    MCTSNode* curr = node;
    while (curr != nullptr) {
        std::lock_guard<std::mutex> lock(curr->node_mutex);
        curr->virtual_loss--;
        curr = curr->parent;
    }
}

// The core loop that calls your Python predictor
void MCTS::search(int iterations, Board& root_board, 
                  const std::function<std::pair<std::vector<float>, RewardVector>(const BoardArray&)>& predictor) {
    
    // 1. Initial expansion of root
    auto root = std::make_unique<MCTSNode>(nullptr, 1.0f);
    auto [init_policy, init_value] = predictor(root_board.get_grid());
    expand(root.get(), root_board, init_policy);

    // 2. Multi-threaded simulation loop (Simplified for structure)
    for (int i = 0; i < iterations; ++i) {
        Board temp_board = root_board;
        MCTSNode* curr = root.get();

        // Selection
        while (curr->is_expanded()) {
            MCTSNode* next = select(curr, temp_board);
            if (!next) break;
            
            // We need to find the Move that leads to 'next'
            // In our map, the key is m.to
            // apply_move({unknown_from, next_key}) -> This needs a small fix in your Board logic!
            curr = next;
        }

        // Expansion & Evaluation (Call the 3080)
        if (!temp_board.is_terminal()) {
            apply_virtual_loss(curr);
            auto [policy, value] = predictor(temp_board.get_grid());
            expand(curr, temp_board, policy);
            remove_virtual_loss(curr);
            backpropagate(curr, value);
        } else {
            backpropagate(curr, temp_board.get_rewards());
        }
    }
}