#!/usr/bin/env bash
#
# YOLO Training Runner with CUDA Library Path
# 
# This script sets the required LD_LIBRARY_PATH for CUDA libraries
# before running the training script.
#

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set CUDA library path from venv
export LD_LIBRARY_PATH="${PROJECT_ROOT}/venv/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"

echo "📚 CUDA Library Path set"
echo "   $LD_LIBRARY_PATH"
echo ""

# Run training script with all arguments passed through
python "${SCRIPT_DIR}/train_yolo.py" "$@"
