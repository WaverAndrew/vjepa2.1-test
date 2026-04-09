"""
Ego4D video loader for V-JEPA 2.1 experiments.

Supports:
  - Sliding-window clip extraction for surprise scoring
  - Ego4D STA v2 annotations for action anticipation evaluation
  - Raw video streaming for long-video summarization

Expected directory layout:
    ego4d_root/
        videos/          # .mp4 files (full-length egocentric videos)
        annotations/
            sta_v2_train.json
            sta_v2_val.json
"""

import json
import math
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

try:
    import decord
    decord.bridge.set_bridge("torch")
    _HAS_DECORD = True
except ImportError:
    _HAS_DECORD = False


# ---------------------------------------------------------------------------
# Standard video transform (matches V-JEPA 2.1 evaluation protocol)
# ---------------------------------------------------------------------------

def make_video_transform(crop_size: int = 384, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    return T.Compose([
        T.Resize(crop_size, antialias=True),
        T.CenterCrop(crop_size),
        T.Normalize(mean=mean, std=std),
    ])


# ---------------------------------------------------------------------------
# Sliding-window dataset for surprise scoring
# ---------------------------------------------------------------------------

class Ego4DSlidingWindow(Dataset):
    """
    Extracts overlapping (context, future) clip pairs from Ego4D videos.

    For each window position t:
      - context: frames [t, t + context_frames)
      - future:  frames [t + context_frames, t + context_frames + future_frames)

    The surprise score is computed by feeding 'context' to the encoder+predictor
    and measuring L1 error against the target-encoder embedding of 'future'.

    Args:
        video_paths: list of .mp4 file paths
        context_frames: number of frames in the context window (default 32 @ 4fps = 8s)
        future_frames: number of frames to predict ahead (default 16 @ 4fps = 4s)
        stride_frames: sliding stride in frames (default 8 = 2s)
        fps: target sampling fps (default 4)
        crop_size: spatial resolution
    """

    def __init__(
        self,
        video_paths: List[str],
        context_frames: int = 32,
        future_frames: int = 16,
        stride_frames: int = 8,
        fps: int = 4,
        crop_size: int = 384,
    ):
        self.context_frames = context_frames
        self.future_frames = future_frames
        self.stride_frames = stride_frames
        self.fps = fps
        self.transform = make_video_transform(crop_size)

        # Build index: (video_path, start_frame_idx)
        self.index: List[Tuple[str, int]] = []
        for vp in video_paths:
            n = self._num_frames(vp)
            total = context_frames + future_frames
            for start in range(0, n - total + 1, stride_frames):
                self.index.append((vp, start))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        video_path, start = self.index[idx]
        total = self.context_frames + self.future_frames

        frames = self._load_frames(video_path, start, total)  # (T, C, H, W) float [0,1]
        frames = self.transform(frames)

        context = frames[:self.context_frames]   # (T_ctx, C, H, W)
        future = frames[self.context_frames:]    # (T_fut, C, H, W)

        return {
            "context": context,
            "future": future,
            "video_path": video_path,
            "start_frame": start,
        }

    # ------------------------------------------------------------------

    def _num_frames(self, video_path: str) -> int:
        if _HAS_DECORD:
            vr = decord.VideoReader(video_path, num_threads=1)
            native_fps = vr.get_avg_fps()
            return int(len(vr) * self.fps / native_fps)
        # Fallback: assume 30fps source
        import av
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            dur = float(stream.duration * stream.time_base)
            return int(dur * self.fps)

    def _load_frames(self, video_path: str, start: int, count: int) -> torch.Tensor:
        """Load 'count' frames starting at 'start' (in target-fps indices)."""
        if _HAS_DECORD:
            vr = decord.VideoReader(video_path, num_threads=4)
            native_fps = vr.get_avg_fps()
            # Map target-fps indices to native indices
            native_indices = [
                min(int((start + i) * native_fps / self.fps), len(vr) - 1)
                for i in range(count)
            ]
            frames = vr.get_batch(native_indices)  # (T, H, W, C) uint8
            frames = frames.float() / 255.0
            frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
        else:
            frames = self._load_frames_av(video_path, start, count)
        return frames

    def _load_frames_av(self, video_path: str, start: int, count: int) -> torch.Tensor:
        import av
        target_times = [(start + i) / self.fps for i in range(count)]
        collected = []
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONREF"
            for frame in container.decode(stream):
                t = float(frame.pts * stream.time_base)
                if len(collected) < count and t >= target_times[len(collected)] - 1e-3:
                    img = torch.from_numpy(frame.to_ndarray(format="rgb24")).float() / 255.0
                    img = img.permute(2, 0, 1)  # (C, H, W)
                    collected.append(img)
                if len(collected) == count:
                    break
        while len(collected) < count:
            collected.append(collected[-1])
        return torch.stack(collected)  # (T, C, H, W)


# ---------------------------------------------------------------------------
# Ego4D STA v2 dataset for action anticipation evaluation
# ---------------------------------------------------------------------------

class Ego4DSTADataset(Dataset):
    """
    Ego4D Short-Term Anticipation v2 dataset.

    Each sample provides:
      - a video clip of 'num_frames' frames ending just before the interaction
      - annotation: noun_class, verb_class, bounding_box, time_to_contact (delta)

    Protocol follows V-JEPA 2.1 paper (Appendix C.4):
      - 16 frames at 2fps, 384px resolution
      - clip ends at the 'last observed frame' V_t
    """

    def __init__(
        self,
        ego4d_root: str,
        split: str = "val",       # 'train' or 'val'
        num_frames: int = 16,
        fps: int = 2,
        crop_size: int = 384,
    ):
        self.ego4d_root = Path(ego4d_root)
        self.num_frames = num_frames
        self.fps = fps
        self.transform = make_video_transform(crop_size)

        ann_file = self.ego4d_root / "annotations" / f"sta_v2_{split}.json"
        with open(ann_file) as f:
            data = json.load(f)

        self.samples = self._parse_annotations(data)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        video_path = str(self.ego4d_root / "videos" / f"{s['video_uid']}.mp4")
        end_time = s["clip_end_sec"]
        start_time = end_time - self.num_frames / self.fps

        frames = self._load_clip(video_path, start_time, end_time)  # (T, C, H, W)
        frames = self.transform(frames)

        return {
            "frames": frames,
            "noun_class": s["noun_class"],
            "verb_class": s["verb_class"],
            "bbox": torch.tensor(s["bbox"], dtype=torch.float32),  # [x1,y1,x2,y2] normalized
            "delta": torch.tensor(s["delta"], dtype=torch.float32),  # time-to-contact (s)
            "video_uid": s["video_uid"],
            "clip_uid": s["clip_uid"],
        }

    def _parse_annotations(self, data: dict) -> List[dict]:
        samples = []
        for clip in data.get("clips", data.get("annotations", [])):
            clip_uid = clip.get("clip_uid", clip.get("uid", ""))
            video_uid = clip.get("video_uid", "")
            for ann in clip.get("annotations", [clip]):
                for frame_ann in ann.get("object_frames", [ann.get("frames", [{}])[0]]):
                    for obj in frame_ann.get("objects", [frame_ann]):
                        samples.append({
                            "video_uid": video_uid,
                            "clip_uid": clip_uid,
                            "clip_end_sec": frame_ann.get("clip_end_sec", 0.0),
                            "noun_class": obj.get("noun_category_id", 0),
                            "verb_class": obj.get("verb_category_id", 0),
                            "bbox": obj.get("box_org", [0, 0, 1, 1]),
                            "delta": obj.get("time_to_contact", 1.0),
                        })
        return samples

    def _load_clip(self, video_path: str, start: float, end: float) -> torch.Tensor:
        target_times = np.linspace(start, end, self.num_frames, endpoint=False).tolist()
        collected = []
        if _HAS_DECORD:
            vr = decord.VideoReader(video_path, num_threads=4)
            native_fps = vr.get_avg_fps()
            indices = [min(int(t * native_fps), len(vr) - 1) for t in target_times]
            frames = vr.get_batch(indices).float() / 255.0
            return frames.permute(0, 3, 1, 2)
        import av
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            for t in target_times:
                container.seek(int(t / stream.time_base), stream=stream)
                for frame in container.decode(stream):
                    img = torch.from_numpy(frame.to_ndarray(format="rgb24")).float() / 255.0
                    collected.append(img.permute(2, 0, 1))
                    break
        while len(collected) < self.num_frames:
            collected.append(collected[-1] if collected else torch.zeros(3, 1, 1))
        return torch.stack(collected[:self.num_frames])
