#!/bin/bash
#SBATCH --job-name=vjepa21_summarize
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/summarize_%j.out
#SBATCH --error=logs/summarize_%j.err
#SBATCH --array=0-9   # process 10 batches in parallel (adjust to your dataset size)

set -e
mkdir -p logs outputs/summaries

module load python/3.10 cuda/12.1 gcc/11.3
conda activate vjepa21

EGO4D_ROOT="/data/ego4d"
MODEL="vit_giant"
OUTPUT_DIR="outputs/summaries"

# SCORER: use 'encoder_distance' for fast experiments, 'prediction_error' for full pipeline
# prediction_error requires --checkpoint pointing to a full .pt with predictor weights
SCORER="encoder_distance"
METHOD="peaks"   # peaks | threshold | budget

python eval/summarize_ego4d.py \
    --ego4d_root "$EGO4D_ROOT" \
    --model "$MODEL" \
    --scorer "$SCORER" \
    --method "$METHOD" \
    --output_dir "$OUTPUT_DIR" \
    --context_frames 32 \
    --future_frames 16 \
    --stride_frames 8 \
    --fps 4 \
    --crop_size 384 \
    --plot_signals \
    --device cuda

# Uncomment below for full prediction-error scorer (needs complete checkpoint):
# CHECKPOINT="/data/checkpoints/vjepa21_vitg.pt"
# python eval/summarize_ego4d.py \
#     --ego4d_root "$EGO4D_ROOT" \
#     --checkpoint "$CHECKPOINT" \
#     --model "$MODEL" \
#     --scorer prediction_error \
#     --method budget \
#     --budget_seconds 300 \
#     --output_dir "$OUTPUT_DIR" \
#     --device cuda
