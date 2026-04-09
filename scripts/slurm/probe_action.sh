#!/bin/bash
#SBATCH --job-name=vjepa21_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/probe_%j.out
#SBATCH --error=logs/probe_%j.err

set -e
mkdir -p logs

module load python/3.10 cuda/12.1 gcc/11.3
conda activate vjepa21

EGO4D_ROOT="/data/ego4d"
MODEL="vit_giant"
OUTPUT_DIR="outputs/probes/action_anticipation"

python eval/probe_action.py \
    --mode train \
    --ego4d_root "$EGO4D_ROOT" \
    --model "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 20 \
    --lr 1e-4 \
    --batch_size 32 \
    --num_workers 16 \
    --num_frames 32 \
    --fps 8 \
    --crop_size 384 \
    --device cuda
