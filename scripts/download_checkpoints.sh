#!/bin/bash
# Download V-JEPA 2.1 checkpoints.
# The easiest route is via torch.hub (no manual download needed for inference).
# Run this script once to pre-cache the hub model to avoid re-downloading on every job.

set -e

CACHE_DIR="${1:-/data/checkpoints/vjepa21_hub_cache}"
mkdir -p "$CACHE_DIR"
export TORCH_HOME="$CACHE_DIR"

echo "Pre-downloading V-JEPA 2.1 checkpoints to $CACHE_DIR ..."

python - <<EOF
import torch

# Distilled ViT-L (300M) — fastest, good for debugging
print("Downloading vit_large (distilled, 300M)...")
m = torch.hub.load("facebookresearch/vjepa2", "vit_large", pretrained=True, trust_repo=True)
del m

# ViT-g (1B) — best quality/speed tradeoff for action understanding
print("Downloading vit_giant (1B)...")
m = torch.hub.load("facebookresearch/vjepa2", "vit_giant", pretrained=True, trust_repo=True)
del m

print("Done. Checkpoints cached in $TORCH_HOME")
EOF
