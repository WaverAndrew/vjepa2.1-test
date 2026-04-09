"""
Notebook-style script: Phase 3 — Surprise scoring on a single Ego4D video.

Shows:
  - The raw surprise signal over time
  - Selected "informative" clips
  - Comparison of encoder-distance vs prediction-error scoring
"""

# %% Setup
import sys; sys.path.insert(0, "..")
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

from src.model.loader import load_encoder_from_hub
encoder = load_encoder_from_hub("vit_giant").to(device)

# %% Encoder-distance scorer (fast — no predictor needed)
from src.surprise.scorer import EncoderDistanceScorer

scorer = EncoderDistanceScorer(encoder, fps=4, device=device)

VIDEO_PATH = "/data/ego4d/videos/your_video.mp4"  # <-- change this

raw_scores = scorer.score_video(
    video_path=VIDEO_PATH,
    clip_frames=16,
    stride_frames=8,
    fps=4,
    crop_size=224,  # use smaller crop for faster iteration
)
print(f"Computed {len(raw_scores)} window scores")

# %% Select and visualize
from src.surprise.summarizer import select_by_peaks, merge_windows, plot_surprise_signal

selected = select_by_peaks(raw_scores, smoothing_window=5, min_distance_windows=4)
segments = merge_windows(selected)

print(f"\nSelected {len(selected)} windows → {len(segments)} segments")
for s, e, sc in segments:
    print(f"  [{s:.1f}s — {e:.1f}s] mean_surprise={sc:.4f}")

plot_surprise_signal(raw_scores, selected=selected, segments=segments,
                     title=f"Surprise Signal — Ego4D")

# %% Show top-5 most surprising clips (PCA visualization)
from src.visualization.pca_features import compute_video_pca, visualize_pca_grid
from src.data.ego4d import Ego4DSlidingWindow

top5 = sorted(raw_scores, key=lambda s: s.score, reverse=True)[:5]
dataset = Ego4DSlidingWindow([VIDEO_PATH], context_frames=16, future_frames=1,
                              stride_frames=8, fps=4, crop_size=224)

for rank, s in enumerate(top5):
    # Find the corresponding window in dataset
    idx = next(i for i, (vp, sf) in enumerate(dataset.index) if sf == s.start_frame)
    frames = dataset[idx]["context"]
    pca_maps, _ = compute_video_pca(encoder, frames, device=device)
    visualize_pca_grid(frames, pca_maps,
                       title=f"Top-{rank+1} Most Surprising (t={s.start_time_sec:.1f}s, score={s.score:.4f})",
                       max_frames=4)
