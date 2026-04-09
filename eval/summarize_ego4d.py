"""
Phase 3+4: Compute surprise scores and summarize long Ego4D videos.

This is the main script for the thesis experiment:
  - Load V-JEPA 2.1 encoder (+ predictor for full surprise scoring)
  - Slide over every Ego4D video in a given directory
  - Compute per-window prediction error (surprise)
  - Select informative clips using peak/threshold/budget selection
  - Export JSON manifests + optionally concatenated summary videos
  - Save a global surprise score CSV for analysis

Usage:
    # Fast mode (encoder-distance scorer, no predictor needed):
    python eval/summarize_ego4d.py \
        --ego4d_root /data/ego4d \
        --model vit_giant \
        --scorer encoder_distance \
        --method peaks \
        --output_dir outputs/summaries \
        --export_video

    # Full mode (prediction error scorer — requires full checkpoint with predictor):
    python eval/summarize_ego4d.py \
        --ego4d_root /data/ego4d \
        --checkpoint /data/checkpoints/vjepa21_vitg.pt \
        --model vit_giant \
        --scorer prediction_error \
        --method budget \
        --budget_seconds 300 \
        --output_dir outputs/summaries
"""

import argparse
import json
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from vjepa21_lib.model.loader import load_encoder_from_hub, load_from_checkpoint
from vjepa21_lib.surprise.scorer import PredictionErrorScorer, EncoderDistanceScorer
from vjepa21_lib.surprise.summarizer import (
    VideoSummarizer, select_by_peaks, select_by_threshold, select_by_budget,
    merge_windows, plot_surprise_signal,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ego4d_root", type=str, required=True,
                   help="Root of Ego4D dataset (must contain videos/ subdir)")
    p.add_argument("--model", type=str, default="vit_giant",
                   choices=["vit_large", "vit_huge", "vit_giant"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Local checkpoint (required for prediction_error scorer)")
    p.add_argument("--scorer", type=str, default="encoder_distance",
                   choices=["encoder_distance", "prediction_error"])
    p.add_argument("--method", type=str, default="peaks",
                   choices=["threshold", "peaks", "budget"])
    p.add_argument("--budget_seconds", type=float, default=120.0)
    p.add_argument("--threshold_percentile", type=float, default=75.0)
    p.add_argument("--output_dir", type=str, default="outputs/summaries")
    p.add_argument("--context_frames", type=int, default=32)
    p.add_argument("--future_frames", type=int, default=16)
    p.add_argument("--stride_frames", type=int, default=8)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--crop_size", type=int, default=384)
    p.add_argument("--max_videos", type=int, default=None,
                   help="Process at most N videos (for debugging)")
    p.add_argument("--export_video", action="store_true",
                   help="Export concatenated summary .mp4 (requires ffmpeg)")
    p.add_argument("--plot_signals", action="store_true",
                   help="Save surprise signal plots for each video")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_scorer(args, encoder, predictor=None, target_encoder=None):
    if args.scorer == "prediction_error":
        assert predictor is not None and target_encoder is not None, \
            "prediction_error scorer requires a full checkpoint with predictor and target_encoder"
        return PredictionErrorScorer(encoder, predictor, target_encoder, fps=args.fps, device=args.device)
    else:
        return EncoderDistanceScorer(encoder, fps=args.fps, device=args.device)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model} | scorer: {args.scorer}")
    if args.checkpoint:
        encoder, predictor, target_encoder = load_from_checkpoint(args.checkpoint, args.model)
    else:
        encoder = load_encoder_from_hub(args.model)
        predictor = target_encoder = None

    scorer = build_scorer(args, encoder, predictor, target_encoder)

    summarizer = VideoSummarizer(
        scorer=scorer,
        method=args.method,
        budget_seconds=args.budget_seconds,
        threshold_percentile=args.threshold_percentile,
        context_frames=args.context_frames,
        future_frames=args.future_frames,
        stride_frames=args.stride_frames,
        fps=args.fps,
        crop_size=args.crop_size,
    )

    # Find all videos
    video_dir = Path(args.ego4d_root) / "videos"
    video_paths = sorted(video_dir.glob("*.mp4"))
    if args.max_videos:
        video_paths = video_paths[:args.max_videos]

    print(f"Processing {len(video_paths)} videos...")

    all_stats = []

    for vp in tqdm(video_paths, desc="Summarizing"):
        try:
            result = summarizer.summarize(
                video_path=str(vp),
                output_dir=str(output_dir / "per_video"),
                export_video=args.export_video,
            )

            all_stats.append({
                "video": vp.name,
                "total_duration_sec": result["total_duration_sec"],
                "summary_duration_sec": result["summary_duration_sec"],
                "compression_ratio": result["compression_ratio"],
                "num_segments": len(result["segments"]),
                "mean_surprise": sum(s["mean_surprise"] for s in result["segments"]) / max(len(result["segments"]), 1),
            })

            if args.plot_signals:
                from vjepa21_lib.surprise.scorer import SurpriseScore
                raw = [SurpriseScore(
                    video_path=str(vp),
                    start_frame=w["start_frame"],
                    end_context_frame=w["start_frame"] + args.context_frames,
                    end_future_frame=w["start_frame"] + args.context_frames + args.future_frames,
                    score=w["score"],
                    start_time_sec=w["start_sec"],
                    end_time_sec=w["start_sec"] + (args.context_frames + args.future_frames) / args.fps,
                ) for w in result["raw_scores"]]

                plot_surprise_signal(
                    raw,
                    title=f"Surprise Signal — {vp.stem}",
                    save_path=str(output_dir / "plots" / f"{vp.stem}_signal.png"),
                )

        except Exception as e:
            print(f"[ERROR] {vp.name}: {e}")

    # Save global stats CSV
    if all_stats:
        df = pd.DataFrame(all_stats)
        csv_path = output_dir / "summary_stats.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nGlobal stats saved: {csv_path}")
        print(df.describe())


if __name__ == "__main__":
    main()
