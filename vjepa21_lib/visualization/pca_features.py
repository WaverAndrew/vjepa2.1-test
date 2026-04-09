"""
PCA visualization of V-JEPA 2.1 patch features (replicates Fig 1, 14, 15 from the paper).

Given a video clip and a frozen encoder, we:
1. Extract patch feature tokens (shape: T*H*W x D)
2. Fit PCA and keep top-3 components
3. Map each component to an RGB channel and overlay on the original frame

This is the primary sanity check that the checkpoint is working correctly.
"""

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA


@torch.no_grad()
def extract_patch_features(
    encoder,
    frames: torch.Tensor,
    device: str = "cuda",
    use_3d_tokenizer: bool = True,
) -> torch.Tensor:
    """
    Extract spatial patch features from a video clip.

    Args:
        encoder: frozen V-JEPA 2.1 encoder (from torch.hub or checkpoint)
        frames: (T, C, H, W) float tensor, normalized
        device: 'cuda' or 'cpu'
        use_3d_tokenizer: True for video (3D conv), False for single images (2D conv)

    Returns:
        features: (N_tokens, D) float tensor on CPU
                  where N_tokens = (T/2) * (H/patch_size) * (W/patch_size)
    """
    frames = frames.to(device)
    # Add batch dim: (1, C, T, H, W) for video input
    if frames.ndim == 4:
        x = frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
    else:
        x = frames

    # Encoder forward — returns (B, N, D)
    features = encoder(x)
    if isinstance(features, (list, tuple)):
        features = features[-1]  # Take last layer output

    return features.squeeze(0).cpu()  # (N, D)


def compute_video_pca(
    encoder,
    frames: torch.Tensor,
    device: str = "cuda",
    n_components: int = 3,
    normalize: bool = True,
) -> tuple:
    """
    Compute PCA on patch features across all frames of a clip.

    Returns:
        pca_maps: (T_out, H_patches, W_patches, 3) float array in [0, 1]
        pca: fitted sklearn PCA object
    """
    feats = extract_patch_features(encoder, frames, device)  # (N, D)

    pca = PCA(n_components=n_components)
    proj = pca.fit_transform(feats.float().numpy())  # (N, 3)

    # Infer spatial/temporal dims
    T_frames = frames.shape[0]
    tubelet_size = 2
    patch_size = 16
    H, W = frames.shape[-2], frames.shape[-1]
    T_out = T_frames // tubelet_size
    H_p = H // patch_size
    W_p = W // patch_size

    proj = proj.reshape(T_out, H_p, W_p, n_components)

    if normalize:
        for c in range(n_components):
            ch = proj[..., c]
            proj[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-6)

    return proj, pca


def visualize_pca_grid(
    frames: torch.Tensor,
    pca_maps: np.ndarray,
    title: str = "V-JEPA 2.1 Dense Features (PCA)",
    save_path: Optional[str] = None,
    max_frames: int = 8,
):
    """
    Plot a grid of (original frame | PCA overlay) side by side.

    frames:   (T, C, H, W) normalized tensor
    pca_maps: (T_out, H_p, W_p, 3) float array from compute_video_pca
    """
    T = frames.shape[0]
    T_out = pca_maps.shape[0]
    tubelet_size = T // T_out
    frame_indices = [i * tubelet_size for i in range(min(T_out, max_frames))]

    n_show = len(frame_indices)
    fig, axes = plt.subplots(2, n_show, figsize=(2.5 * n_show, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # Unnormalize frames for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    frames_display = (frames.cpu() * std + mean).clamp(0, 1)

    for col, fi in enumerate(frame_indices):
        ti = fi // tubelet_size  # PCA map index

        # Original frame
        ax = axes[0, col] if n_show > 1 else axes[0]
        img = frames_display[fi].permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"t={fi}", fontsize=8)

        # PCA feature map
        ax = axes[1, col] if n_show > 1 else axes[1]
        pca_img = pca_maps[ti]  # (H_p, W_p, 3)
        # Upsample to frame resolution for visual clarity
        pca_tensor = torch.from_numpy(pca_img).permute(2, 0, 1).unsqueeze(0)  # (1,3,H_p,W_p)
        H, W = img.shape[:2]
        pca_up = F.interpolate(pca_tensor, size=(H, W), mode="bilinear", align_corners=False)
        pca_up = pca_up.squeeze(0).permute(1, 2, 0).numpy()
        ax.imshow(pca_up)
        ax.axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9) if n_show > 1 else axes[0].set_ylabel("Original")
    axes[1, 0].set_ylabel("PCA Features", fontsize=9) if n_show > 1 else axes[1].set_ylabel("PCA Features")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def visualize_temporal_consistency(
    encoder,
    frames_list: List[torch.Tensor],
    labels: List[str],
    device: str = "cuda",
    save_path: Optional[str] = None,
):
    """
    Compare PCA feature consistency across multiple models or clips.
    frames_list: list of (T, C, H, W) tensors
    labels: display names for each
    """
    n = len(frames_list)
    fig, axes = plt.subplots(n, 4, figsize=(12, 3 * n))

    for row, (frames, label) in enumerate(zip(frames_list, labels)):
        pca_maps, _ = compute_video_pca(encoder, frames, device)
        T = frames.shape[0]
        tubelet_size = T // pca_maps.shape[0]

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frames_display = (frames.cpu() * std + mean).clamp(0, 1)

        for col in range(4):
            t_idx = col * (T // 4)
            ti = t_idx // tubelet_size

            ax = axes[row, col] if n > 1 else axes[col]
            pca_tensor = torch.from_numpy(pca_maps[ti]).permute(2, 0, 1).unsqueeze(0)
            H, W = frames.shape[-2], frames.shape[-1]
            pca_up = F.interpolate(pca_tensor, size=(H, W), mode="bilinear", align_corners=False)
            pca_up = pca_up.squeeze(0).permute(1, 2, 0).numpy()
            ax.imshow(pca_up)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(label, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
