"""
Phase 1: Visualize V-JEPA 2.1 dense patch features on Ego4D clips.
Replicates the PCA visualizations from Fig 1, 14, 15 of the paper.

Usage:
    python eval/visualize_features.py \
        --video_path /path/to/ego4d/videos/sample.mp4 \
        --model vit_giant \
        --output_dir outputs/visualizations \
        --num_frames 32 \
        --crop_size 384
"""

import argparse
from pathlib import Path

import torch

from src.model.loader import load_encoder_from_hub
from src.data.ego4d import Ego4DSlidingWindow
from src.visualization.pca_features import compute_video_pca, visualize_pca_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--model", type=str, default="vit_giant",
                   choices=["vit_large", "vit_huge", "vit_giant", "vit_bigG"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Local checkpoint path (optional; uses torch.hub if not set)")
    p.add_argument("--output_dir", type=str, default="outputs/visualizations")
    p.add_argument("--num_frames", type=int, default=32,
                   help="Number of frames to load (default 32 @ 4fps = 8s)")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=384)
    p.add_argument("--start_time_sec", type=float, default=0.0,
                   help="Start time in video (seconds) for the clip to visualize")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading {args.model} ...")
    if args.checkpoint:
        from src.model.loader import load_from_checkpoint
        encoder, _, _ = load_from_checkpoint(args.checkpoint, args.model)
    else:
        encoder = load_encoder_from_hub(args.model)
    encoder = encoder.to(args.device)

    # Load a clip
    from src.data.ego4d import Ego4DSlidingWindow
    dataset = Ego4DSlidingWindow(
        video_paths=[args.video_path],
        context_frames=args.num_frames,
        future_frames=1,           # we only need context frames here
        stride_frames=args.num_frames,
        fps=args.fps,
        crop_size=args.crop_size,
    )

    # Find the window closest to the requested start time
    target_start_frame = int(args.start_time_sec * args.fps)
    best_idx = min(range(len(dataset)), key=lambda i: abs(dataset.index[i][1] - target_start_frame))
    batch = dataset[best_idx]
    frames = batch["context"]  # (T, C, H, W)

    print(f"Loaded clip: {frames.shape} | start_frame={batch['start_frame']}")

    # Compute PCA
    print("Computing PCA features...")
    pca_maps, pca = compute_video_pca(encoder, frames, device=args.device)

    # Visualize
    video_name = Path(args.video_path).stem
    save_path = str(output_dir / f"{video_name}_pca_features.png")
    visualize_pca_grid(frames, pca_maps, title=f"V-JEPA 2.1 Dense Features — {video_name}", save_path=save_path)

    # Also save explained variance for sanity check
    print(f"\nPCA explained variance: {pca.explained_variance_ratio_}")
    print(f"Visualization saved to: {save_path}")


if __name__ == "__main__":
    main()
