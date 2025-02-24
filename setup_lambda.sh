#!/bin/bash

# Exit on error
set -e

# Clone repository if not already present
if [ ! -d "ubi" ]; then
    git clone https://github.com/tamimeur/ubi.git
    cd ubi/antibody_design
else
    cd ubi/antibody_design
    git pull origin main
fi

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install GPU-enabled dependencies
pip install -r requirements-gpu.txt

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_PROJECT="antibody-affinity"

# Create necessary directories
mkdir -p checkpoints
mkdir -p data

# Run training
echo "Starting training..."
python src/train_affinity.py
