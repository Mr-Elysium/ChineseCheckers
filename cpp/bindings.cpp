#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "board.hpp"
#include "mcts.hpp"
#include "move_gen.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cc_core, m) {
    m.doc() = "Chinese Checkers Engine Core";

    // 1. Bind the Move struct
    py::class_<Move>(m, "Move")
        .def(py::init<int, int>())
        .def_readwrite("from_idx", &Move::from_idx)
        .def_readwrite("to_idx", &Move::to_idx)
        .def("__repr__", [](const Move &a) {
            return "<Move from " + std::to_string(a.from_idx) + " to " + std::to_string(a.to_idx) + ">";
        });

    // 2. Bind the Board class (Optimized get_grid)
    py::class_<Board>(m, "Board")
        .def(py::init<int>(), py::arg("num_players") = 2)
        .def("apply_move", &Board::apply_move)
        .def("get_grid", [](const Board &b) {
            // Shape: 121 elements (sequential star positions)
            // Strides: 1 byte per element (int8_t)
            // Pointer: direct to the C++ internal array
            // Base: we cast the Board object 'b' so Python knows it owns the memory
            return py::array_t<int8_t>(
                {121},               // shape
                {sizeof(int8_t)},    // strides
                b.get_grid_ptr(),    // ptr
                py::cast(b)          // base object
            );
        }, py::return_value_policy::reference_internal) // Crucial policy for safety
        .def("get_current_player", &Board::get_current_player)
        .def("is_terminal", &Board::is_terminal)
        .def("get_rewards", &Board::get_rewards);

    // 3. Bind the MoveGen class (Static methods)
    py::class_<MoveGen>(m, "MoveGen")
        .def_static("initialize", &MoveGen::initialize)
        .def_static("get_legal_moves", &MoveGen::get_legal_moves);

    // 4. Bind the MCTS class
    py::class_<MCTS>(m, "MCTS")
        .def(py::init<float>(), py::arg("c_puct") = 1.41f)
        .def("search", &MCTS::search)
        .def("get_best_move", &MCTS::get_best_move)
        .def("get_best_move_and_reuse", &MCTS::get_best_move_and_reuse)
        .def("get_visit_counts", &MCTS::get_visit_counts)
        .def("clear_tree", &MCTS::clear_tree);
}