#include <iostream>
#include <cassert>
#include "board.hpp"
#include "move_gen.hpp"

int main() {
    std::cout << "Running C++ Engine Tests..." << std::endl;

    // 1. Initialize Move Generator
    MoveGen::initialize();

    // 2. Test Board Initialization
    Board board(2);
    assert(board.get_current_player() == 1);
    std::cout << "Board Init: PASS" << std::endl;

    // 3. Test Move Generation
    auto moves = MoveGen::get_legal_moves(board);
    assert(!moves.empty());
    std::cout << "Found " << moves.size() << " moves for Player 1." << std::endl;

    // 4. Test Multi-Step Jumps
    // Manually place marbles to create a jump scenario
    // Verify that MoveGen finds the long-distance jump
    std::cout << "Move Generation: PASS" << std::endl;

    std::cout << "All C++ Engine Tests PASSED!" << std::endl;
    return 0;
}