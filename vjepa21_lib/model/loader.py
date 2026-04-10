"""
Load V-JEPA 2.1 encoder from locally downloaded .pt weights.

Bypasses torch.hub.load entirely (its URLs point to localhost:8300, broken).
Uses the same model construction logic as the official hubconf.py.

V-JEPA 2.1 models:
    vjepa2_1_vit_base_384     -- ViT-B  (80M,  distilled)
    vjepa2_1_vit_large_384    -- ViT-L  (300M, distilled) <- fast tests
    vjepa2_1_vit_giant_384    -- ViT-g  (1B)              <- main experiments
    vjepa2_1_vit_gigantic_384 -- ViT-G  (2B)              <- best quality

Requires:
    1. git clone https://github.com/facebookresearch/vjepa2 $VJEPA2_DIR
    2. bash scripts/download_checkpoints.sh
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

VJEPA2_DIR = "/scratch/3206024/vjepa2_official"
WEIGHTS_DIR = "/home/3206024/vjepa2.1-test/weights"

# model_name → (arch_builder_name, weight_filename, embed_dim)
MODEL_REGISTRY = {
    "vjepa2_1_vit_base_384": {
        "arch": "vit_base",
        "file": "vjepa2_1_vitb_dist_vitG_384.pt",
        "embed_dim": 768,
    },
    "vjepa2_1_vit_large_384": {
        "arch": "vit_large",
        "file": "vjepa2_1_vitl_dist_vitG_384.pt",
        "embed_dim": 1024,
    },
    "vjepa2_1_vit_giant_384": {
        "arch": "vit_giant",
        "file": "vjepa2_1_vitg_384.pt",
        "embed_dim": 1408,
    },
    "vjepa2_1_vit_gigantic_384": {
        "arch": "vit_gigantic",
        "file": "vjepa2_1_vitG_384.pt",
        "embed_dim": 1664,
    },
}


def _ensure_vjepa2_importable():
    if VJEPA2_DIR not in sys.path:
        sys.path.insert(0, VJEPA2_DIR)


def _clean_backbone_key(state_dict):
    """Strip 'module.' and 'backbone.' prefixes — matches official hubconf."""
    cleaned = {}
    for key, val in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = val
    return cleaned


def load_encoder_from_hub(
    model_name: str = "vjepa2_1_vit_giant_384",
    pretrained: bool = True,
    num_frames: int = 16,
) -> nn.Module:
    """
    Load a V-JEPA 2.1 encoder.

    Mirrors the official _make_vjepa2_1_model() logic:
    - Builds the encoder from app.vjepa_2_1.models.vision_transformer
    - Loads state_dict from the 'target_encoder' key
    - Strips 'module.' and 'backbone.' prefixes
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {list(MODEL_REGISTRY)}")

    info = MODEL_REGISTRY[model_name]
    _ensure_vjepa2_importable()

    # Import from the same path as the official hubconf
    from app.vjepa_2_1.models import vision_transformer as vit_encoder

    encoder_kwargs = dict(
        patch_size=16,
        img_size=(384, 384),
        num_frames=num_frames,
        tubelet_size=2,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )

    # Build encoder architecture
    encoder = vit_encoder.__dict__[info["arch"]](**encoder_kwargs)

    if pretrained:
        weight_path = Path(WEIGHTS_DIR) / info["file"]
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Weights not found: {weight_path}\n"
                "Run:  bash scripts/download_checkpoints.sh"
            )
        print(f"Loading weights: {weight_path}")
        ckpt = torch.load(str(weight_path), map_location="cpu", weights_only=False)

        # Official checkpoint stores encoder under 'target_encoder' key
        if "target_encoder" in ckpt:
            state_dict = _clean_backbone_key(ckpt["target_encoder"])
        elif "encoder" in ckpt:
            state_dict = _clean_backbone_key(ckpt["encoder"])
        else:
            state_dict = _clean_backbone_key(ckpt)

        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing[:3]}...")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:3]}...")
        if not missing and not unexpected:
            print(f"  All keys matched.")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def get_encoder_dim(model_name: str) -> int:
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Valid: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name]["embed_dim"]
