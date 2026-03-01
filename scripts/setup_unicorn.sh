#!/bin/bash
#
# One-time setup for the Unicorn machine (or any Linux box with NVIDIA GPUs).
# Creates venv, installs dependencies, verifies GPU detection, runs tests.
#
# Usage:
#   bash scripts/setup_unicorn.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================="
echo "Setting up VAN project on $(hostname)"
echo "============================================="

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "venv already exists"
fi

source venv/bin/activate

# Install PyTorch with CUDA 12.8 and other dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy matplotlib sympy pytest

# Verify GPU detection
echo ""
echo "Verifying GPU setup..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    n = torch.cuda.device_count()
    print(f'GPU count: {n}')
    for i in range(n):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f'  GPU {i}: {name} ({mem:.1f} GB)')
    # Verify float64 support
    x = torch.randn(10, dtype=torch.float64, device='cuda')
    y = x @ x
    print(f'float64 on CUDA: OK (dot product = {y.item():.6f})')
else:
    print('WARNING: No CUDA GPUs detected')
"

# Run test suite
echo ""
echo "Running test suite..."
pytest tests/ -v

echo ""
echo "============================================="
echo "Setup complete!"
echo "============================================="
