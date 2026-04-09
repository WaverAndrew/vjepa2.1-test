#!/bin/bash
# Run on HPC login node to set up the environment.
# Adjust module names to match your HPC's module system.

set -e

# --- HPC modules (edit to match your cluster) ---
module load python/3.10
module load cuda/12.1
module load gcc/11.3

# --- Create conda env ---
conda create -n vjepa21 python=3.10 -y
conda activate vjepa21

# --- Install PyTorch (adjust cuda version) ---
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# --- Clone official V-JEPA 2 repo (needed for src/) ---
git clone https://github.com/facebookresearch/vjepa2.git vjepa2_official
pip install -e vjepa2_official/

# --- Install project deps ---
pip install -r requirements.txt

echo "Environment ready. Activate with: conda activate vjepa21"
