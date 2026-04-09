"""
Notebook-style script: Phase 1 — Feature visualization.
Run as a script or convert to .ipynb with: jupytext --to notebook this_file.py

Step-by-step walk-through of V-JEPA 2.1 dense feature extraction and PCA visualization.
"""

# %% [markdown]
# ## Phase 1: V-JEPA 2.1 Dense Feature Visualization
#
# This notebook replicates Fig 1, 14, 15 from the paper.
# We load the ViT-G encoder via torch.hub, extract patch tokens from an Ego4D clip,
# and visualize the top-3 PCA components as RGB feature maps.

# %% Setup
import sys; sys.path.insert(0, "..")
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# %% Load encoder via torch.hub
from src.model.loader import load_encoder_from_hub
encoder = load_encoder_from_hub("vit_giant")  # change to vit_large for faster iteration
encoder = encoder.to(device)
print(f"Encoder loaded. Parameters: {sum(p.numel() for p in encoder.parameters()) / 1e6:.0f}M")

# %% Load a sample Ego4D clip
from src.data.ego4d import Ego4DSlidingWindow

VIDEO_PATH = "/data/ego4d/videos/your_video.mp4"  # <-- change this
dataset = Ego4DSlidingWindow(
    video_paths=[VIDEO_PATH],
    context_frames=32,
    future_frames=1,
    stride_frames=32,
    fps=4,
    crop_size=384,
)

batch = dataset[0]
frames = batch["context"]  # (32, 3, 384, 384)
print(f"Loaded clip: {frames.shape}, start_frame={batch['start_frame']}")

# %% Compute PCA features
from src.visualization.pca_features import compute_video_pca, visualize_pca_grid

pca_maps, pca = compute_video_pca(encoder, frames, device=device)
print(f"PCA maps shape: {pca_maps.shape}")  # (T/2, H_p, W_p, 3)
print(f"Explained variance: {pca.explained_variance_ratio_}")

# %% Visualize
visualize_pca_grid(frames, pca_maps, title="V-JEPA 2.1 Dense Features (Ego4D)", max_frames=8)

# %% Compare V-JEPA 2.1 feature consistency across multiple clips
# Grab 4 clips from the same video and show how features evolve
clips = [dataset[i]["context"] for i in range(min(4, len(dataset)))]
for i, clip in enumerate(clips):
    pca_m, _ = compute_video_pca(encoder, clip, device=device)
    visualize_pca_grid(clip, pca_m, title=f"Clip {i} (t={dataset.index[i][1]/4:.1f}s)")
