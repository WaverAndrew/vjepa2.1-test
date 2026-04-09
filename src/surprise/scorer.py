"""
Prediction Error Surprise Scorer — the core of the video summarization system.

Core idea (from V-JEPA 2.1's training objective):
  The predictor P_φ is trained to predict the latent representation of masked
  future patches from visible context patches. At inference, we repurpose this
  as a surprise signal:

      surprise(t) = L1( P_φ(E_θ(context_t), Δ_future), sg(E_θ̄(future_t)) )

  High surprise = the model's world model failed to predict what happened next
               = the clip contains novel / informationally rich content.

  Low surprise  = the model correctly predicted what came next
               = the clip is redundant / routine.

This is the principled, model-native way to measure information content —
we're directly probing the model's internal world model.

Alternative (lighter) scorer using only the encoder:
  cosine_surprise(t) = 1 - cosine_sim(mean_pool(E(context_t)), mean_pool(E(future_t)))
  Works without the predictor but is less sensitive to fine-grained dynamics.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------

def build_future_mask_tokens(
    future_frames: torch.Tensor,
    patch_size: int = 16,
    tubelet_size: int = 2,
    encoder_dim: int = 1408,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Build learnable mask tokens with spatio-temporal positional info
    for the future frames (tokens that the predictor needs to fill in).

    Returns: (1, N_future, encoder_dim) — initialized to zeros with position info
    """
    T, C, H, W = future_frames.shape
    T_out = T // tubelet_size
    H_p = H // patch_size
    W_p = W // patch_size
    N = T_out * H_p * W_p
    # Mask tokens: zeros + positional encoding added by the predictor internally
    return torch.zeros(1, N, encoder_dim, device=device)


# ---------------------------------------------------------------------------
# Main Scorer
# ---------------------------------------------------------------------------

@dataclass
class SurpriseScore:
    """Per-window surprise score result."""
    video_path: str
    start_frame: int
    end_context_frame: int
    end_future_frame: int
    score: float          # higher = more surprising / informative
    start_time_sec: float
    end_time_sec: float


class PredictionErrorScorer:
    """
    Computes prediction-error-based surprise scores using the full V-JEPA 2.1
    encoder + predictor.

    For each (context, future) window pair:
      1. Encode context with E_θ   → context tokens
      2. Forward predictor P_φ     → predicted future token representations
      3. Encode future with E_θ̄   → target future token representations
      4. Score = mean L1(predicted, target)

    Args:
        encoder: context encoder E_θ (frozen)
        predictor: predictor P_φ (frozen)
        target_encoder: EMA encoder E_θ̄ (frozen)
        fps: video fps used for time calculations
        device: 'cuda' or 'cpu'
    """

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        target_encoder: nn.Module,
        fps: int = 4,
        device: str = "cuda",
    ):
        self.encoder = encoder.to(device).eval()
        self.predictor = predictor.to(device).eval()
        self.target_encoder = target_encoder.to(device).eval()
        self.fps = fps
        self.device = device

    @torch.no_grad()
    def score_window(
        self,
        context: torch.Tensor,
        future: torch.Tensor,
    ) -> float:
        """
        Score a single (context, future) window pair.

        context: (T_ctx, C, H, W) normalized float tensor
        future:  (T_fut, C, H, W) normalized float tensor

        Returns: scalar surprise score (L1 prediction error, averaged over tokens)
        """
        # Reshape to (1, C, T, H, W) for video encoder
        ctx = context.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # (1,C,T_ctx,H,W)
        fut = future.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)   # (1,C,T_fut,H,W)

        # 1. Encode context → context tokens
        ctx_tokens = self.encoder(ctx)
        if isinstance(ctx_tokens, (list, tuple)):
            ctx_tokens = ctx_tokens[-1]
        # ctx_tokens: (1, N_ctx, D)

        # 2. Get target representation of future frames
        fut_tokens = self.target_encoder(fut)
        if isinstance(fut_tokens, (list, tuple)):
            fut_tokens = fut_tokens[-1]
        # fut_tokens: (1, N_fut, D)

        # 3. Build mask token positions for future frames
        #    The predictor expects: [ctx_tokens | mask_tokens] → predictions for mask positions
        T_fut, C, H, W = future.shape
        patch_size = 16
        tubelet_size = 2
        T_out = T_fut // tubelet_size
        H_p = H // patch_size
        W_p = W // patch_size
        N_fut = T_out * H_p * W_p

        # Mask tokens: zeros (the predictor adds positional encoding internally)
        mask_tokens = torch.zeros(1, N_fut, ctx_tokens.shape[-1], device=self.device)

        # Concatenate: the predictor processes context + mask tokens jointly
        full_sequence = torch.cat([ctx_tokens, mask_tokens], dim=1)

        # 4. Predictor forward → predictions for all positions
        #    We only care about the predictions at mask token positions (last N_fut)
        pred_all = self.predictor(full_sequence)
        if isinstance(pred_all, (list, tuple)):
            pred_all = pred_all[-1]
        pred_future = pred_all[:, -N_fut:, :]  # (1, N_fut, D)

        # 5. Surprise = mean L1 between predicted and actual future representations
        # Normalize both to unit sphere (optional but stabilizes scores)
        pred_norm = F.normalize(pred_future, dim=-1)
        tgt_norm = F.normalize(fut_tokens, dim=-1)

        score = F.l1_loss(pred_norm, tgt_norm).item()
        return score

    def score_video(
        self,
        video_path: str,
        context_frames: int = 32,
        future_frames: int = 16,
        stride_frames: int = 8,
        fps: int = 4,
        crop_size: int = 384,
    ) -> List[SurpriseScore]:
        """
        Score an entire video with a sliding window.

        Returns list of SurpriseScore objects, one per window position.
        """
        from src.data.ego4d import Ego4DSlidingWindow
        from torch.utils.data import DataLoader

        dataset = Ego4DSlidingWindow(
            video_paths=[video_path],
            context_frames=context_frames,
            future_frames=future_frames,
            stride_frames=stride_frames,
            fps=fps,
            crop_size=crop_size,
        )
        loader = DataLoader(dataset, batch_size=1, num_workers=2, pin_memory=True)

        results = []
        for batch in loader:
            context = batch["context"][0]   # (T_ctx, C, H, W)
            future = batch["future"][0]     # (T_fut, C, H, W)
            start = batch["start_frame"].item()

            score = self.score_window(context, future)

            results.append(SurpriseScore(
                video_path=video_path,
                start_frame=start,
                end_context_frame=start + context_frames,
                end_future_frame=start + context_frames + future_frames,
                score=score,
                start_time_sec=start / fps,
                end_time_sec=(start + context_frames + future_frames) / fps,
            ))

        return results


class EncoderDistanceScorer:
    """
    Lightweight surprise scorer using only the encoder (no predictor needed).

    Computes 1 - cosine_similarity between consecutive clip embeddings.
    Fast and useful for initial experiments / ablations.

    score(t) = 1 - cos_sim(pool(E(clip_t)), pool(E(clip_{t+1})))
    """

    def __init__(self, encoder: nn.Module, fps: int = 4, device: str = "cuda"):
        self.encoder = encoder.to(device).eval()
        self.fps = fps
        self.device = device

    @torch.no_grad()
    def score_video(
        self,
        video_path: str,
        clip_frames: int = 16,
        stride_frames: int = 8,
        fps: int = 4,
        crop_size: int = 384,
    ) -> List[SurpriseScore]:
        from src.data.ego4d import Ego4DSlidingWindow
        from torch.utils.data import DataLoader

        # Use 2x clips: context = clip_t, "future" = clip_{t+stride}
        dataset = Ego4DSlidingWindow(
            video_paths=[video_path],
            context_frames=clip_frames,
            future_frames=clip_frames,
            stride_frames=stride_frames,
            fps=fps,
            crop_size=crop_size,
        )
        loader = DataLoader(dataset, batch_size=4, num_workers=4, pin_memory=True)

        results = []
        for batch in loader:
            context = batch["context"].to(self.device)   # (B, T, C, H, W)
            future = batch["future"].to(self.device)

            B = context.shape[0]
            # Reshape to (B, C, T, H, W)
            ctx_vid = context.permute(0, 2, 1, 3, 4)
            fut_vid = future.permute(0, 2, 1, 3, 4)

            ctx_feats = self.encoder(ctx_vid)
            fut_feats = self.encoder(fut_vid)
            if isinstance(ctx_feats, (list, tuple)):
                ctx_feats, fut_feats = ctx_feats[-1], fut_feats[-1]

            # Mean pool over tokens
            ctx_pool = ctx_feats.mean(dim=1)  # (B, D)
            fut_pool = fut_feats.mean(dim=1)

            sim = F.cosine_similarity(ctx_pool, fut_pool, dim=-1)  # (B,)
            scores = (1 - sim).cpu().tolist()

            for i, score in enumerate(scores):
                start = batch["start_frame"][i].item()
                results.append(SurpriseScore(
                    video_path=video_path,
                    start_frame=start,
                    end_context_frame=start + clip_frames,
                    end_future_frame=start + 2 * clip_frames,
                    score=score,
                    start_time_sec=start / fps,
                    end_time_sec=(start + 2 * clip_frames) / fps,
                ))

        return results
