#!/bin/bash

# Exit on error
set -e

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3.9-venv nvidia-cuda-toolkit

# Clone repository
echo "Cloning repository..."
git clone https://github.com/tamimeur/ubi.git
cd ubi/antibody_design

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements-gpu.txt

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use single GPU
export WANDB_PROJECT="antibody-affinity"

# Create necessary directories
mkdir -p checkpoints
mkdir -p data

# Verify CUDA setup
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0))"

# Run training
echo "Starting training..."
python src/train_affinity.py
