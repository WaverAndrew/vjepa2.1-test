#!/bin/bash
# Download V-JEPA 2.1 checkpoints into the project weights/ subfolder.
# Weights land at: /home/3206024/vjepa2.1-test/weights/hub/checkpoints/
# Run once on the login node (needs internet).

set -e

REPO_DIR="/home/3206024/vjepa2.1-test"
WEIGHTS_DIR="$REPO_DIR/weights"
VJEPA2_DIR="/scratch/3206024/vjepa2_official"

echo "=== Step 1: Clone official vjepa2 repo to $VJEPA2_DIR ==="
if [ -d "$VJEPA2_DIR" ]; then
    echo "Already cloned, pulling latest..."
    git -C "$VJEPA2_DIR" pull
else
    git clone https://github.com/facebookresearch/vjepa2.git "$VJEPA2_DIR"
fi

echo ""
echo "=== Step 2: Add vjepa2 to PYTHONPATH (no install needed) ==="
export PYTHONPATH="$VJEPA2_DIR:${PYTHONPATH}"

echo ""
echo "=== Step 3: Download V-JEPA 2.1 weights into $WEIGHTS_DIR ==="
mkdir -p "$WEIGHTS_DIR"
export TORCH_HOME="$WEIGHTS_DIR"

cd "$VJEPA2_DIR"
python - <<EOF
import torch

# V-JEPA 2.1 ViT-L/384 (300M distilled) — fastest, use for first tests
print("Downloading vjepa2_1_vit_large_384 (300M)...")
m = torch.hub.load(
    "$VJEPA2_DIR", "vjepa2_1_vit_large_384",
    pretrained=True, source="local", trust_repo=True
)
del m
print("  vit_large done.")

# V-JEPA 2.1 ViT-g/384 (1B) — main model for experiments
print("Downloading vjepa2_1_vit_giant_384 (1B)...")
m = torch.hub.load(
    "$VJEPA2_DIR", "vjepa2_1_vit_giant_384",
    pretrained=True, source="local", trust_repo=True
)
del m
print("  vit_giant done.")

print(f"\nWeights saved under: $WEIGHTS_DIR/hub/checkpoints/")
EOF

echo ""
echo "=== Done ==="
echo "  VJEPA2_DIR  = $VJEPA2_DIR"
echo "  WEIGHTS_DIR = $WEIGHTS_DIR"
