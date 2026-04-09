#!/bin/bash
#SBATCH --job-name=vjepa21_probe
#SBATCH --account=3206024
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err

set -e
mkdir -p logs

module load cuda/12.1
module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv

export TORCH_HOME=/scratch/3206024/torch_hub_cache
mkdir -p "$TORCH_HOME"

cd /home/3206024/vjepa2.1-test

# Requires Ego4D STA v2 annotations — use this script only when Ego4D is available.
# For HD-EPIC there are no action labels, so probing is done via surprise scoring instead.
echo "Action probe requires Ego4D STA v2 annotations."
echo "For HD-EPIC, run summarize.sh instead to generate surprise-scored summaries."
