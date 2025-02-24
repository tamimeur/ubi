#!/bin/bash

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install GPU-enabled dependencies
pip install -r requirements-gpu.txt

# Set CUDA device if multiple GPUs are available
export CUDA_VISIBLE_DEVICES=0

# Run training script
python src/train_affinity.py
