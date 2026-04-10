"""
Visualize PCA features on multiple clips from different HD-EPIC videos.

Usage:
    python eval/visualize_multi.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import setup_paths  # noqa

from pathlib import Path
import torch

from vjepa21_lib.model.loader import load_encoder_from_hub
from vjepa21_lib.data.hd_epic import list_videos
from vjepa21_lib.data.ego4d import Ego4DSlidingWindow
from vjepa21_lib.visualization.pca_features import compute_video_pca, visualize_pca_grid

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading vjepa2_1_vit_large_384 ...")
encoder = load_encoder_from_hub("vjepa2_1_vit_large_384", num_frames=16).to(device)

# Pick one video from each participant
all_videos = list_videos("/scratch/HD-EPIC")
participants_seen = set()
selected = []
for v in all_videos:
    pid = v.parent.name
    if pid not in participants_seen:
        participants_seen.add(pid)
        selected.append(v)
    if len(selected) >= 5:
        break

print(f"Selected {len(selected)} videos from participants: {[v.parent.name for v in selected]}")

output_dir = Path("outputs/visualizations/multi")
output_dir.mkdir(parents=True, exist_ok=True)

for vid_path in selected:
    pid = vid_path.parent.name
    print(f"\nProcessing {pid}/{vid_path.name} ...")

    dataset = Ego4DSlidingWindow(
        video_paths=[str(vid_path)],
        context_frames=16,
        future_frames=1,
        stride_frames=16,
        fps=4,
        crop_size=384,
    )

    if len(dataset) == 0:
        print(f"  Skipping — no frames extracted")
        continue

    # Sample from the middle of the video
    mid_idx = len(dataset) // 2
    batch = dataset[mid_idx]
    frames = batch["context"]
    start_sec = batch["start_frame"] / 4.0

    print(f"  Clip: {frames.shape}, t={start_sec:.1f}s (window {mid_idx}/{len(dataset)})")

    pca_maps, pca = compute_video_pca(encoder, frames, device=device)

    save_path = str(output_dir / f"{pid}_{vid_path.stem}_pca.png")
    visualize_pca_grid(
        frames, pca_maps,
        title=f"V-JEPA 2.1 — {pid}/{vid_path.stem} (t={start_sec:.0f}s)",
        save_path=save_path,
        max_frames=8,
    )
    print(f"  Variance explained: {pca.explained_variance_ratio_}")

print(f"\nAll done. Check: {output_dir}/")
