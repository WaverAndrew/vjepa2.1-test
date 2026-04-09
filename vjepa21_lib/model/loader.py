"""
Load V-JEPA 2.1 encoder from locally downloaded .pt weights.

Bypasses torch.hub.load entirely (its URLs point to localhost:8300, broken).
Instead we: import the ViT builder from the cloned vjepa2 repo, construct the
model, and load the .pt state dict directly.

V-JEPA 2.1 models:
    vjepa2_1_vit_base_384     -- ViT-B  (80M,  distilled)
    vjepa2_1_vit_large_384    -- ViT-L  (300M, distilled) <- fast tests
    vjepa2_1_vit_giant_384    -- ViT-g  (1B)              <- main experiments
    vjepa2_1_vit_gigantic_384 -- ViT-G  (2B)              <- best quality

Requires:
    1. git clone https://github.com/facebookresearch/vjepa2 $VJEPA2_DIR
    2. bash scripts/download_checkpoints.sh
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

VJEPA2_DIR = os.environ.get("VJEPA2_DIR", "/scratch/3206024/vjepa2_official")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/home/3206024/vjepa2.1-test/weights")

# model_name → (weight filename, builder name, embed dim)
MODEL_REGISTRY = {
    "vjepa2_1_vit_base_384": {
        "file": "vjepa2_1_vitb_dist_vitG_384.pt",
        "builder": "vit_base",
        "embed_dim": 768,
    },
    "vjepa2_1_vit_large_384": {
        "file": "vjepa2_1_vitl_dist_vitG_384.pt",
        "builder": "vit_large",
        "embed_dim": 1024,
    },
    "vjepa2_1_vit_giant_384": {
        "file": "vjepa2_1_vitg_384.pt",
        "builder": "vit_giant",
        "embed_dim": 1408,
    },
    "vjepa2_1_vit_gigantic_384": {
        "file": "vjepa2_1_vitG_384.pt",
        "builder": "vit_gigantic",
        "embed_dim": 1664,
    },
}


def _ensure_vjepa2_importable():
    """Add the cloned vjepa2 repo to sys.path."""
    repo = VJEPA2_DIR
    if not Path(repo).exists():
        raise FileNotFoundError(
            f"vjepa2 repo not found at {repo}.\n"
            "Run:  git clone https://github.com/facebookresearch/vjepa2 {repo}\n"
            "or set VJEPA2_DIR env var."
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)


def load_encoder_from_hub(
    model_name: str = "vjepa2_1_vit_giant_384",
    pretrained: bool = True,
) -> nn.Module:
    """
    Load a V-JEPA 2.1 encoder by importing the architecture from the vjepa2 repo
    and loading the .pt weights directly (bypasses broken torch.hub URLs).

    Returns the encoder in eval mode with all parameters frozen.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {list(MODEL_REGISTRY)}")

    info = MODEL_REGISTRY[model_name]
    _ensure_vjepa2_importable()

    # Import the model builder from the cloned vjepa2 repo
    from src.models.vision_transformer import (
        vit_base, vit_large, vit_giant, vit_gigantic,
    )
    builders = {
        "vit_base": vit_base,
        "vit_large": vit_large,
        "vit_giant": vit_giant,
        "vit_gigantic": vit_gigantic,
    }
    encoder = builders[info["builder"]]()

    if pretrained:
        weight_path = Path(WEIGHTS_DIR) / info["file"]
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {weight_path}\n"
                "Run:  bash scripts/download_checkpoints.sh"
            )
        print(f"Loading weights: {weight_path}")
        state_dict = torch.load(str(weight_path), map_location="cpu")

        # Handle different checkpoint key formats
        if "encoder" in state_dict:
            state_dict = state_dict["encoder"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        # Strip DDP 'module.' prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def get_encoder_dim(model_name: str) -> int:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name]["embed_dim"]
