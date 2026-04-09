#!/bin/bash
#SBATCH --job-name=vjepa21_summarize
#SBATCH --account=3206024
#SBATCH --partition=gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/summarize_%j.out
#SBATCH --error=logs/summarize_%j.err

set -e
mkdir -p logs

module load cuda/12.1
module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv

export VJEPA2_DIR=/scratch/3206024/vjepa2_official
export WEIGHTS_DIR=/home/3206024/vjepa2.1-test/weights
export TORCH_HOME="$WEIGHTS_DIR"

cd /home/3206024/vjepa2.1-test

python eval/summarize_hd_epic.py \
    --hd_epic_root /scratch/HD-EPIC \
    --participants P01 \
    --model vjepa2_1_vit_giant_384 \
    --scorer encoder_distance \
    --method peaks \
    --output_dir outputs/summaries \
    --context_frames 32 \
    --future_frames 16 \
    --stride_frames 8 \
    --fps 4 \
    --crop_size 384 \
    --plot_signals \
    --device cuda
