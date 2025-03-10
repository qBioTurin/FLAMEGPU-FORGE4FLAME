#!/bin/bash

echo "===== System Check for GPU & Development Tools ====="

# Check for NVIDIA GPU
echo -n "Checking for NVIDIA GPU... "
if lspci | grep -i nvidia > /dev/null; then
    echo "✅ Found"
else
    echo "❌ No NVIDIA GPU detected"
    exit 1
fi

# Check for NVIDIA drivers
echo -n "Checking for NVIDIA drivers... "
if command -v nvidia-smi > /dev/null; then
    echo "✅ Installed"
    nvidia-smi
else
    echo "❌ NVIDIA drivers NOT installed"
    exit 1
fi

# Check for CUDA installation
echo -n "Checking for CUDA... "
if command -v nvcc > /dev/null then
    CUDA_VERSION=$(nvcc --version | grep -oP "release \K[0-9]+(\.[0-9]+)*")
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    if (( CUDA_MAJOR >= 11 )); then
        echo "✅ Installed (Version: $CUDA_VERSION)"
    else
        echo "❌ CUDA version is too old ($CUDA_VERSION). Require ≥ 11.0"
        exit 1
    fi
else
    echo "❌ CUDA is NOT installed"
    exit 1
fi

# Check Compute Capability
echo -n "Checking GPU Compute Capability (≥ 3.5)... "
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | awk -F. '{ print $1$2 }')
if [[ "$CC" -ge 35 ]]; then
    echo "✅ Compute Capability: $CC"
else
    echo "❌ Compute Capability ($CC) is too low. Require ≥ 3.5"
    exit 1
fi

# Check CMake version
echo -n "Checking for CMake (≥ 3.18)... "
if command -v cmake > /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -n1 | grep -oP "[0-9]+\.[0-9]+\.[0-9]+")
    CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
    CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)
    if (( CMAKE_MAJOR > 3 || (CMAKE_MAJOR == 3 && CMAKE_MINOR >= 18) )); then
        echo "✅ Installed (Version: $CMAKE_VERSION)"
    else
        echo "❌ CMake version ($CMAKE_VERSION) is too old. Require ≥ 3.18"
        exit 1
    fi
else
    echo "❌ CMake is NOT installed"
    exit 1
fi

# Check for GCC and make
echo -n "Checking for GCC (≥ 8.1)... "
if command -v gcc > /dev/null; then
    GCC_VERSION=$(gcc -dumpversion)
    GCC_MAJOR=$(echo "$GCC_VERSION" | cut -d. -f1)
    GCC_MINOR=$(echo "$GCC_VERSION" | cut -d. -f2)
    if (( GCC_MAJOR > 8 || (GCC_MAJOR == 8 && GCC_MINOR >= 1) )); then
        echo "✅ Installed (Version: $GCC_VERSION)"
    else
        echo "❌ GCC version ($GCC_VERSION) is too old. Require ≥ 8.1"
        exit 1
    fi
else
    echo "❌ GCC is NOT installed"
    exit 1
fi

echo -n "Checking for make... "
if command -v make > /dev/null; then
    echo "✅ Installed"
else
    echo "❌ make is NOT installed"
    exit 1
fi

# Check for Python installation
echo -n "Checking for Python... "
if command -v python3 > /dev/null; then
    PYTHON_VERSION=$(python3 --version | grep -oP "[0-9]+\.[0-9]+\.[0-9]+")
    echo "✅ Installed (Version: $PYTHON_VERSION)"
else
    echo "❌ Python is NOT installed"
    exit 1
fi

echo "✅ All checks passed successfully!"
exit 0
