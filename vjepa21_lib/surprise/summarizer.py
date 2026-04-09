"""
Video Summarization from Surprise Scores.

Given per-window surprise scores from the scorer, this module selects
the most informative clips from a long egocentric video stream.

Strategy:
  1. Smooth the surprise signal (optional) to reduce shot-boundary noise
  2. Select clips via one of three methods:
     a. Threshold: keep all windows above a percentile threshold
     b. Peak detection: find local maxima (surprise "events")
     c. Knapsack: maximize total surprise subject to budget (output duration)
  3. Merge overlapping selected windows
  4. Export the summary as a new video file or a timestamped manifest

Output: a JSON manifest and optionally a concatenated summary .mp4
"""

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks, savgol_filter

from vjepa21_lib.surprise.scorer import SurpriseScore


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def select_by_threshold(
    scores: List[SurpriseScore],
    percentile: float = 75.0,
) -> List[SurpriseScore]:
    """Keep windows whose surprise score exceeds the given percentile."""
    values = np.array([s.score for s in scores])
    thresh = np.percentile(values, percentile)
    return [s for s in scores if s.score >= thresh]


def select_by_peaks(
    scores: List[SurpriseScore],
    smoothing_window: int = 5,
    min_distance_windows: int = 4,
    prominence_percentile: float = 25.0,
) -> List[SurpriseScore]:
    """
    Find local surprise peaks (events) using scipy peak detection.

    This models the intuition that "something meaningful happened" corresponds
    to a local maximum in the surprise signal — e.g., a person picks up an
    object, starts a new activity, or enters a new scene.

    Args:
        smoothing_window: Savitzky-Golay filter window (set 1 to disable)
        min_distance_windows: minimum gap between peaks in window units
        prominence_percentile: minimum peak prominence (as percentile of signal range)
    """
    values = np.array([s.score for s in scores])

    if smoothing_window > 1 and len(values) >= smoothing_window:
        values_smooth = savgol_filter(values, smoothing_window, polyorder=2)
    else:
        values_smooth = values

    signal_range = values_smooth.max() - values_smooth.min() + 1e-8
    min_prominence = signal_range * prominence_percentile / 100.0

    peaks, properties = find_peaks(
        values_smooth,
        distance=min_distance_windows,
        prominence=min_prominence,
    )

    return [scores[i] for i in peaks]


def select_by_budget(
    scores: List[SurpriseScore],
    budget_seconds: float = 120.0,
    fps: int = 4,
    context_frames: int = 32,
) -> List[SurpriseScore]:
    """
    Greedy knapsack: maximize total surprise within a time budget.

    Sorts windows by score (descending), picks non-overlapping ones
    until budget is exhausted.
    """
    window_duration = context_frames / fps
    max_windows = int(budget_seconds / window_duration)

    sorted_scores = sorted(scores, key=lambda s: s.score, reverse=True)
    selected = []
    selected_ranges = []

    for s in sorted_scores:
        if len(selected) >= max_windows:
            break
        # Check overlap with already-selected windows
        overlap = any(
            not (s.end_future_frame <= r[0] or s.start_frame >= r[1])
            for r in selected_ranges
        )
        if not overlap:
            selected.append(s)
            selected_ranges.append((s.start_frame, s.end_future_frame))

    return sorted(selected, key=lambda s: s.start_frame)


# ---------------------------------------------------------------------------
# Merge overlapping windows
# ---------------------------------------------------------------------------

def merge_windows(
    scores: List[SurpriseScore],
    gap_frames: int = 8,
) -> List[Tuple[float, float, float]]:
    """
    Merge selected windows that are close together into contiguous segments.

    Returns: list of (start_sec, end_sec, mean_score) tuples
    """
    if not scores:
        return []

    sorted_scores = sorted(scores, key=lambda s: s.start_frame)
    segments = []
    seg_start = sorted_scores[0].start_time_sec
    seg_end = sorted_scores[0].end_time_sec
    seg_scores = [sorted_scores[0].score]

    for s in sorted_scores[1:]:
        if s.start_frame <= sorted_scores[sorted_scores.index(s) - 1].end_future_frame + gap_frames:
            seg_end = s.end_time_sec
            seg_scores.append(s.score)
        else:
            segments.append((seg_start, seg_end, float(np.mean(seg_scores))))
            seg_start = s.start_time_sec
            seg_end = s.end_time_sec
            seg_scores = [s.score]

    segments.append((seg_start, seg_end, float(np.mean(seg_scores))))
    return segments


# ---------------------------------------------------------------------------
# Main summarizer
# ---------------------------------------------------------------------------

class VideoSummarizer:
    """
    Full summarization pipeline for long egocentric video streams.

    Usage:
        summarizer = VideoSummarizer(scorer, method='peaks')
        summary = summarizer.summarize(video_path, output_dir)
    """

    def __init__(
        self,
        scorer,  # PredictionErrorScorer or EncoderDistanceScorer
        method: str = "peaks",         # 'threshold', 'peaks', 'budget'
        budget_seconds: float = 120.0, # for 'budget' method
        threshold_percentile: float = 75.0,
        smoothing_window: int = 5,
        context_frames: int = 32,
        future_frames: int = 16,
        stride_frames: int = 8,
        fps: int = 4,
        crop_size: int = 384,
    ):
        self.scorer = scorer
        self.method = method
        self.budget_seconds = budget_seconds
        self.threshold_percentile = threshold_percentile
        self.smoothing_window = smoothing_window
        self.context_frames = context_frames
        self.future_frames = future_frames
        self.stride_frames = stride_frames
        self.fps = fps
        self.crop_size = crop_size

    def summarize(
        self,
        video_path: str,
        output_dir: str,
        export_video: bool = False,
    ) -> dict:
        """
        Summarize a single video.

        Returns a summary dict with:
          - raw_scores: per-window scores
          - selected_windows: chosen windows
          - segments: merged time segments
          - stats: summary statistics
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        video_name = Path(video_path).stem

        print(f"[Summarizer] Scoring {video_path} ...")
        raw_scores = self.scorer.score_video(
            video_path=video_path,
            context_frames=self.context_frames,
            future_frames=self.future_frames,
            stride_frames=self.stride_frames,
            fps=self.fps,
            crop_size=self.crop_size,
        )

        print(f"[Summarizer] Computed {len(raw_scores)} window scores. Selecting clips...")
        if self.method == "threshold":
            selected = select_by_threshold(raw_scores, self.threshold_percentile)
        elif self.method == "peaks":
            selected = select_by_peaks(raw_scores, self.smoothing_window)
        elif self.method == "budget":
            selected = select_by_budget(raw_scores, self.budget_seconds, self.fps, self.context_frames)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        segments = merge_windows(selected)

        total_input_sec = (raw_scores[-1].end_time_sec - raw_scores[0].start_time_sec) if raw_scores else 0
        total_summary_sec = sum(e - s for s, e, _ in segments)

        result = {
            "video_path": video_path,
            "method": self.method,
            "total_duration_sec": total_input_sec,
            "summary_duration_sec": total_summary_sec,
            "compression_ratio": total_summary_sec / max(total_input_sec, 1),
            "num_windows": len(raw_scores),
            "num_selected_windows": len(selected),
            "segments": [
                {"start_sec": s, "end_sec": e, "mean_surprise": sc}
                for s, e, sc in segments
            ],
            "raw_scores": [
                {"start_frame": s.start_frame, "score": s.score, "start_sec": s.start_time_sec}
                for s in raw_scores
            ],
        }

        manifest_path = output_dir / f"{video_name}_summary.json"
        with open(manifest_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[Summarizer] Manifest saved: {manifest_path}")

        if export_video:
            self._export_summary_video(video_path, segments, output_dir / f"{video_name}_summary.mp4")

        return result

    def _export_summary_video(
        self,
        video_path: str,
        segments: List[Tuple[float, float, float]],
        output_path: Path,
    ):
        """Concatenate selected segments into a summary video using ffmpeg."""
        import subprocess, tempfile, os

        # Write a concat list
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for (start, end, _) in segments:
                # Write as ffmpeg segment filter input
                f.write(f"file '{video_path}'\n")
                f.write(f"inpoint {start:.3f}\n")
                f.write(f"outpoint {end:.3f}\n")
            concat_file = f.name

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.unlink(concat_file)
        print(f"[Summarizer] Summary video saved: {output_path}")


# ---------------------------------------------------------------------------
# Visualization of the surprise signal
# ---------------------------------------------------------------------------

def plot_surprise_signal(
    scores: List[SurpriseScore],
    selected: Optional[List[SurpriseScore]] = None,
    segments: Optional[List[Tuple]] = None,
    title: str = "Surprise Signal",
    save_path: Optional[str] = None,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    times = [s.start_time_sec for s in scores]
    values = [s.score for s in scores]

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(times, values, color="steelblue", linewidth=1.0, label="Surprise score")
    ax.fill_between(times, values, alpha=0.2, color="steelblue")

    if selected:
        sel_times = [s.start_time_sec for s in selected]
        sel_vals = [s.score for s in selected]
        ax.scatter(sel_times, sel_vals, color="red", zorder=5, s=40, label="Selected windows")

    if segments:
        for (start, end, sc) in segments:
            ax.axvspan(start, end, alpha=0.15, color="green")

    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Prediction Error (Surprise)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close()
