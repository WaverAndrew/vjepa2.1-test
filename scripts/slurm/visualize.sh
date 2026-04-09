#!/bin/bash
#SBATCH --job-name=vjepa21_vis
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/vis_%j.out
#SBATCH --error=logs/vis_%j.err
# Adjust partition to your HPC cluster (e.g., --partition=gpu)

set -e
mkdir -p logs

# Load modules (edit to match your HPC)
module load python/3.10 cuda/12.1 gcc/11.3
conda activate vjepa21

# --- Edit these paths ---
VIDEO_PATH="/data/ego4d/videos/sample.mp4"
OUTPUT_DIR="outputs/visualizations"
MODEL="vit_giant"   # vit_large / vit_huge / vit_giant

python eval/visualize_features.py \
    --video_path "$VIDEO_PATH" \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --num_frames 32 \
    --fps 4 \
    --crop_size 384 \
    --device cuda
