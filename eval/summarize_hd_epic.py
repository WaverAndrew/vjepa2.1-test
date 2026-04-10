"""
Surprise-based video summarization on HD-EPIC.

Usage:
    # Fast test on one participant, one video:
    python eval/summarize_hd_epic.py \
        --hd_epic_root /scratch/HD-EPIC \
        --participants P01 \
        --max_videos 1 \
        --model vit_giant \
        --scorer encoder_distance \
        --method peaks \
        --output_dir outputs/summaries \
        --plot_signals \
        --device cuda

    # Full run on all participants:
    python eval/summarize_hd_epic.py \
        --hd_epic_root /scratch/HD-EPIC \
        --model vit_giant \
        --scorer encoder_distance \
        --method peaks \
        --output_dir outputs/summaries \
        --plot_signals \
        --device cuda
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import setup_paths  # noqa

import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from vjepa21_lib.model.loader import load_encoder_from_hub, load_from_checkpoint
from vjepa21_lib.data.hd_epic import list_videos
from vjepa21_lib.surprise.scorer import EncoderDistanceScorer, PredictionErrorScorer
from vjepa21_lib.surprise.summarizer import (
    VideoSummarizer, select_by_peaks, merge_windows, plot_surprise_signal,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hd_epic_root", type=str, default="/scratch/HD-EPIC")
    p.add_argument("--participants", type=str, nargs="+", default=None,
                   help="Participant IDs to process (default: all). E.g. P01 P04")
    p.add_argument("--model", type=str, default="vjepa2_1_vit_giant_384",
                   choices=["vjepa2_1_vit_base_384", "vjepa2_1_vit_large_384",
                            "vjepa2_1_vit_giant_384", "vjepa2_1_vit_gigantic_384"])
    p.add_argument("--checkpoint", type=str, default=None)
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
    p.add_argument("--max_videos", type=int, default=None)
    p.add_argument("--plot_signals", action="store_true")
    p.add_argument("--export_video", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    (output_dir / "per_video").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    # Load encoder
    print(f"Loading {args.model} ...")
    if args.checkpoint:
        encoder, predictor, target_encoder = load_from_checkpoint(args.checkpoint, args.model)
    else:
        encoder = load_encoder_from_hub(args.model)
        predictor = target_encoder = None

    if args.scorer == "prediction_error":
        assert predictor is not None, "--checkpoint required for prediction_error scorer"
        scorer = PredictionErrorScorer(encoder, predictor, target_encoder,
                                        fps=args.fps, device=args.device)
    else:
        scorer = EncoderDistanceScorer(encoder, fps=args.fps, device=args.device)

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

    video_paths = list_videos(args.hd_epic_root, args.participants)
    if args.max_videos:
        video_paths = video_paths[:args.max_videos]

    print(f"Processing {len(video_paths)} HD-EPIC videos...")

    all_stats = []
    for vp in tqdm(video_paths, desc="Summarizing"):
        try:
            result = summarizer.summarize(
                video_path=str(vp),
                output_dir=str(output_dir / "per_video"),
                export_video=args.export_video,
            )

            if args.plot_signals and result["raw_scores"]:
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
                    title=f"Surprise — {vp.parent.name}/{vp.stem}",
                    save_path=str(output_dir / "plots" / f"{vp.parent.name}_{vp.stem}_signal.png"),
                )

            all_stats.append({
                "participant": vp.parent.name,
                "video": vp.name,
                "total_duration_sec": result["total_duration_sec"],
                "summary_duration_sec": result["summary_duration_sec"],
                "compression_ratio": result["compression_ratio"],
                "num_segments": len(result["segments"]),
            })

        except Exception as e:
            print(f"[ERROR] {vp.name}: {e}")

    if all_stats:
        df = pd.DataFrame(all_stats)
        csv_path = output_dir / "summary_stats.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nStats saved to {csv_path}")
        print(df.to_string())


if __name__ == "__main__":
    main()
