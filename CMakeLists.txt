cmake_minimum_required(VERSION 3.14)
project(ChineseCheckersAZ VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization for Zen-architecture Ryzen
# --- Optimization: Handle Mac vs. PC ---
if(APPLE)
    # Force ARM64 architecture for Apple Silicon Macs
    # This overrides Rosetta if your terminal is lying to CMake
    set(CMAKE_OSX_ARCHITECTURES "arm64")
    
    add_compile_options(-O3 -ffast-math)
    message(STATUS "macOS detected: FORCING arm64 (Apple Silicon) build")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    # For your Ryzen PC later
    add_compile_options(-O3 -march=native -ffast-math)
    message(STATUS "Linux/Windows detected: Using native architecture optimization")
else()
    add_compile_options(-O3)
endif()

find_package(Threads REQUIRED)

# --- Dependency: pybind11 ---
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# --- 1. Build the Python Module ---
# This looks into the cpp/ folder and executes the CMakeLists.txt there
add_subdirectory(cpp)

# --- 2. Build the C++ Test Executable ---
enable_testing()

# Note the paths: we need the engine logic + the test file
add_executable(cpp_test 
    tests/test_cpp.cpp 
    cpp/src/board.cpp 
    cpp/src/move_gen.cpp 
    cpp/src/mcts.cpp
)

# Tell the compiler where to find the .hpp files
target_include_directories(cpp_test PRIVATE cpp/include)
target_link_libraries(cpp_test PRIVATE Threads::Threads)