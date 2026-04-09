"""
Load V-JEPA 2.1 encoder from the locally cloned vjepa2 repo.

V-JEPA 2.1 hub model names (source: hubconf.py):
    vjepa2_1_vit_base_384    -- ViT-B  (80M,  distilled)
    vjepa2_1_vit_large_384   -- ViT-L  (300M, distilled) ← fast tests
    vjepa2_1_vit_giant_384   -- ViT-g  (1B)              ← main experiments
    vjepa2_1_vit_gigantic_384-- ViT-G  (2B)              ← best quality

NOTE: these are V-JEPA 2.1 names. V-JEPA 2 names (vjepa2_vit_*) are different
      and load different checkpoints — do not mix them up.

Requires the repo to be cloned first:
    bash scripts/download_checkpoints.sh
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

# Set by download_checkpoints.sh; override with env vars if needed
VJEPA2_DIR = os.environ.get("VJEPA2_DIR", "/scratch/3206024/vjepa2_official")
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/home/3206024/vjepa2.1-test/weights")

# Valid V-JEPA 2.1 model names → encoder embedding dim
MODEL_DIMS = {
    "vjepa2_1_vit_base_384":     768,
    "vjepa2_1_vit_large_384":   1024,
    "vjepa2_1_vit_giant_384":   1408,
    "vjepa2_1_vit_gigantic_384": 1664,
}


def load_encoder_from_hub(
    model_name: str = "vjepa2_1_vit_giant_384",
    pretrained: bool = True,
) -> nn.Module:
    """
    Load a V-JEPA 2.1 encoder from the locally cloned repo.

    Args:
        model_name: one of the vjepa2_1_* names listed in MODEL_DIMS above
        pretrained: download and load pretrained weights

    Returns:
        encoder in eval mode with all parameters frozen
    """
    if model_name not in MODEL_DIMS:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid V-JEPA 2.1 names: {list(MODEL_DIMS)}"
        )

    repo = VJEPA2_DIR
    if not Path(repo).exists():
        raise FileNotFoundError(
            f"vjepa2 repo not found at {repo}.\n"
            "Run:  bash scripts/download_checkpoints.sh\n"
            "or set VJEPA2_DIR env var to your clone path."
        )

    # Add repo root to path so 'src.hub' is importable (no pip install needed)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    os.environ["TORCH_HOME"] = WEIGHTS_DIR

    encoder = torch.hub.load(
        repo,
        model_name,
        pretrained=pretrained,
        source="local",
        trust_repo=True,
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def get_encoder_dim(model_name: str) -> int:
    if model_name not in MODEL_DIMS:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {list(MODEL_DIMS)}")
    return MODEL_DIMS[model_name]
