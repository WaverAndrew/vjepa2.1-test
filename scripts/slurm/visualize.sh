#!/bin/bash
#SBATCH --job-name=vjepa21_vis
#SBATCH --account=3206024
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/vis_%j.out
#SBATCH --error=logs/vis_%j.err

set -e
mkdir -p logs

module load cuda/12.1
module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv

# Cache checkpoints on scratch so compute nodes don't re-download
export VJEPA2_DIR=/scratch/3206024/vjepa2_official
export WEIGHTS_DIR=/home/3206024/vjepa2.1-test/weights
export TORCH_HOME="$WEIGHTS_DIR"

cd /home/3206024/vjepa2.1-test

# Pick one video from P01 for the sanity check
VIDEO_PATH="/scratch/HD-EPIC/Videos/P01/P01-20240202-161948.mp4"
OUTPUT_DIR="outputs/visualizations"

python eval/visualize_features.py \
    --video_path "$VIDEO_PATH" \
    --model vjepa2_1_vit_giant_384 \
    --output_dir "$OUTPUT_DIR" \
    --num_frames 32 \
    --fps 4 \
    --crop_size 384 \
    --device cuda

echo "Done. Check $OUTPUT_DIR for the PCA feature map PNG."
