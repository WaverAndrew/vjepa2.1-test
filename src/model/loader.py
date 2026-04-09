"""
Load V-JEPA 2.1 encoder and predictor from a checkpoint.

The official checkpoints are loaded via torch.hub from facebookresearch/vjepa2.
We expose the encoder (frozen) and the predictor (needed for surprise scoring).

Usage:
    encoder, predictor, target_encoder = load_vjepa21(cfg)
"""

import torch
import torch.nn as nn
from pathlib import Path


def load_encoder_from_hub(model_name: str = "vit_giant", pretrained: bool = True):
    """
    Load V-JEPA 2.1 encoder via torch.hub.

    model_name options (from facebookresearch/vjepa2):
        'vit_large'  -- ViT-L (300M, distilled from ViT-G)
        'vit_huge'   -- ViT-H (600M)
        'vit_giant'  -- ViT-g (1B)
        'vit_bigG'   -- ViT-G (2B) -- V-JEPA 2.1 flagship

    Returns the encoder in eval mode with frozen parameters.
    """
    encoder = torch.hub.load(
        "facebookresearch/vjepa2",
        model_name,
        pretrained=pretrained,
        trust_repo=True,
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def load_from_checkpoint(checkpoint_path: str, model_name: str = "vit_giant"):
    """
    Load encoder + predictor from a local .pt checkpoint file.

    The checkpoint is expected to contain keys:
        'encoder'   -- state dict for the encoder (context / EMA)
        'predictor' -- state dict for the predictor

    Adjust key names if your checkpoint differs.
    """
    from vjepa2.src.models.vision_transformer import vit_giant, vit_large, vit_huge
    from vjepa2.src.models.predictor import VisionTransformerPredictor

    _builders = {
        "vit_large": vit_large,
        "vit_huge": vit_huge,
        "vit_giant": vit_giant,
    }
    assert model_name in _builders, f"Unknown model: {model_name}"

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    encoder = _builders[model_name]()
    _load_state(encoder, ckpt, key="encoder")
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Target encoder (EMA copy) — same architecture
    target_encoder = _builders[model_name]()
    _load_state(target_encoder, ckpt, key="target_encoder")
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad_(False)

    # Predictor
    predictor = None
    if "predictor" in ckpt:
        predictor = VisionTransformerPredictor(
            num_patches=None,  # will be set dynamically
            embed_dim=1024,    # ViT-g encoder output dim
            predictor_embed_dim=384,
            depth=24,
            num_heads=16,
        )
        _load_state(predictor, ckpt, key="predictor")
        predictor.eval()
        for p in predictor.parameters():
            p.requires_grad_(False)

    return encoder, predictor, target_encoder


def _load_state(model: nn.Module, ckpt: dict, key: str):
    """Load a sub-state-dict from a checkpoint, handling 'module.' prefixes."""
    if key not in ckpt:
        raise KeyError(f"Key '{key}' not found in checkpoint. Available: {list(ckpt.keys())}")
    state = ckpt[key]
    # Strip DDP 'module.' prefix if present
    state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[loader] {key} missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[loader] {key} unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
