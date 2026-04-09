"""
HD-EPIC video loader.

Dataset structure:
    /scratch/HD-EPIC/Videos/
        P01/  P04/  P05/  P06/  P07/  P08/  P09/
            P01-20240202-161948.mp4
            P01-20240202-161948_vrs_to_mp4_log.json   (ignored)
            P01-20240203-121517_mp4_to_vrs_time_ns.csv (ignored)
            ...

Videos are long egocentric recordings. No action annotations are used
for Phase 1 (visualization) and Phase 3 (surprise scoring).
"""

from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from vjepa21_lib.data.ego4d import make_video_transform

HD_EPIC_ROOT = "/scratch/HD-EPIC"


def list_videos(root: str = HD_EPIC_ROOT, participants: Optional[List[str]] = None) -> List[Path]:
    """
    Return sorted list of all .mp4 video paths in HD-EPIC.

    Args:
        root: path to HD-EPIC root (must contain Videos/ subdir)
        participants: filter by participant IDs e.g. ['P01', 'P04'].
                      None = all participants.
    """
    video_dir = Path(root) / "Videos"
    all_mp4 = sorted(video_dir.glob("**/*.mp4"))
    if participants:
        all_mp4 = [p for p in all_mp4 if p.parent.name in participants]
    return all_mp4


class HDEpicSlidingWindow(Dataset):
    """
    Sliding-window clip extractor for HD-EPIC videos.

    Identical interface to Ego4DSlidingWindow but scans HD-EPIC directory
    structure automatically. Used for:
      - Phase 1: feature visualization
      - Phase 3: surprise scoring / video summarization

    Args:
        root: HD-EPIC root directory
        participants: list of participant IDs to include (None = all)
        context_frames: frames in context window (default 32 @ 4fps = 8s)
        future_frames: frames to score ahead (default 16 @ 4fps = 4s)
        stride_frames: sliding stride (default 8 = 2s)
        fps: target sampling fps
        crop_size: spatial resolution
        max_videos: cap number of videos (useful for quick debugging)
    """

    def __init__(
        self,
        root: str = HD_EPIC_ROOT,
        participants: Optional[List[str]] = None,
        context_frames: int = 32,
        future_frames: int = 16,
        stride_frames: int = 8,
        fps: int = 4,
        crop_size: int = 384,
        max_videos: Optional[int] = None,
    ):
        from vjepa21_lib.data.ego4d import Ego4DSlidingWindow

        video_paths = list_videos(root, participants)
        if max_videos:
            video_paths = video_paths[:max_videos]

        if not video_paths:
            raise FileNotFoundError(f"No .mp4 files found under {root}/Videos/")

        print(f"[HDEpic] Found {len(video_paths)} videos across "
              f"{len(set(p.parent.name for p in video_paths))} participants")

        # Reuse Ego4DSlidingWindow logic — same sliding-window mechanics
        self._inner = Ego4DSlidingWindow(
            video_paths=[str(p) for p in video_paths],
            context_frames=context_frames,
            future_frames=future_frames,
            stride_frames=stride_frames,
            fps=fps,
            crop_size=crop_size,
        )
        self.video_paths = video_paths

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, idx):
        return self._inner[idx]

    @property
    def index(self):
        return self._inner.index
