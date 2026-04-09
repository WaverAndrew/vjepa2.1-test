#!/bin/bash
# Download V-JEPA 2.1 checkpoints directly from Meta's CDN.
# The official repo's torch.hub URLs point to localhost:8300 (broken),
# so we download the .pt files manually with wget.
#
# Weights land at: /home/3206024/vjepa2.1-test/weights/
# Run once on the login node (needs internet).

set -e

REPO_DIR="/home/3206024/vjepa2.1-test"
WEIGHTS_DIR="$REPO_DIR/weights"
VJEPA2_DIR="/scratch/3206024/vjepa2_official"

BASE_URL="https://dl.fbaipublicfiles.com/vjepa2"

echo "=== Step 1: Clone official vjepa2 repo ==="
if [ -d "$VJEPA2_DIR" ]; then
    echo "Already cloned at $VJEPA2_DIR"
else
    git clone https://github.com/facebookresearch/vjepa2.git "$VJEPA2_DIR"
fi

echo ""
echo "=== Step 2: Download V-JEPA 2.1 weights into $WEIGHTS_DIR ==="
mkdir -p "$WEIGHTS_DIR"

# ViT-L/384 (300M, distilled) — fast testing
echo "Downloading vjepa2_1_vit_large_384 (300M)..."
wget -c -q --show-progress -O "$WEIGHTS_DIR/vjepa2_1_vitl_dist_vitG_384.pt" \
    "$BASE_URL/vjepa2_1_vitl_dist_vitG_384.pt"

# ViT-g/384 (1B) — main experiments
echo "Downloading vjepa2_1_vit_giant_384 (1B)..."
wget -c -q --show-progress -O "$WEIGHTS_DIR/vjepa2_1_vitg_384.pt" \
    "$BASE_URL/vjepa2_1_vitg_384.pt"

echo ""
echo "=== Done ==="
echo "Weights saved at:"
ls -lh "$WEIGHTS_DIR"/*.pt
echo ""
echo "Set these env vars in your scripts:"
echo "  export VJEPA2_DIR=$VJEPA2_DIR"
echo "  export WEIGHTS_DIR=$WEIGHTS_DIR"

# Available checkpoints (uncomment to download):
# wget -c -O "$WEIGHTS_DIR/vjepa2_1_vitb_dist_vitG_384.pt"  "$BASE_URL/vjepa2_1_vitb_dist_vitG_384.pt"  # ViT-B (80M)
# wget -c -O "$WEIGHTS_DIR/vjepa2_1_vitG_384.pt"            "$BASE_URL/vjepa2_1_vitG_384.pt"            # ViT-G (2B)
