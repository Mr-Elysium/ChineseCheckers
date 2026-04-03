# This creates the cc_core shared library
pybind11_add_module(cc_core 
    src/board.cpp 
    src/move_gen.cpp 
    src/mcts.cpp
    bindings.cpp
)

# Use CMAKE_CURRENT_SOURCE_DIR so it finds the local include/ folder
target_include_directories(cc_core PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link threads for parallel MCTS
target_link_libraries(cc_core PRIVATE Threads::Threads)