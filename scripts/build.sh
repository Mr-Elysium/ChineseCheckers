#!/bin/bash
set -e

echo "--- Detecting Environment ---"
# Determine number of cores for parallel build
if [[ "$OSTYPE" == "darwin"* ]]; then
    THREADS=$(sysctl -n hw.ncpu)
    echo "Running on macOS (ARM). Using $THREADS threads."
else
    THREADS=$(nproc)
    echo "Running on Linux/Windows. Using $THREADS threads."
fi

mkdir -p build
cd build

echo "--- Running CMake Configuration ---"
cmake ..

echo "--- Compiling with $THREADS threads ---"
make -j$THREADS

cd ..

# Find and copy the module
LIB_PATH=$(find build/cpp -name "cc_core*.so" -o -name "cc_core*.pyd" -o -name "cc_core*.dylib")

if [ -f "$LIB_PATH" ]; then
    mkdir -p python
    cp "$LIB_PATH" python/cc_core.so # Rename to .so so Python finds it easily
    echo "Successfully moved cc_core to python/ directory."
else
    echo "Error: Build finished but cc_core module was not found."
    exit 1
fi