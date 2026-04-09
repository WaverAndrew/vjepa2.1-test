"""
Attentive probe for action understanding on top of frozen V-JEPA 2.1 features.

Architecture (from Appendix C.5 and C.6 of the paper):
  - 4 transformer blocks (first 3: self-attention, last 1: cross-attention)
  - Cross-attention in the last block uses a set of learnable query tokens
  - For action anticipation: 3 query tokens (verb, noun, action)
  - Followed by a final linear classifier per query token

For video classification (SSv2, K400, Diving-48):
  - 1 query token
  - Tokens extracted from multiple encoder layers (last + 3 intermediate)

Training:
  - Frozen encoder, only probe parameters are updated
  - Focal loss for action anticipation (alpha=0.25, gamma=2.0)
  - Standard cross-entropy for classification
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, Nq, D = q.shape
        B, Nkv, _ = kv.shape
        queries = self.q_proj(q).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv_out = self.kv_proj(kv).reshape(B, Nkv, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        keys, values = kv_out.unbind(0)
        attn = (queries @ keys.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ values).transpose(1, 2).reshape(B, Nq, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, mlp_ratio: float = 4.0, cross_attention: bool = False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)

        if cross_attention:
            self.attn = MultiHeadCrossAttention(dim, num_heads)
        else:
            self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.cross_attention = cross_attention

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
        if self.cross_attention:
            assert context is not None
            x = x + self.attn(self.norm1(x), self.norm2(context))
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm_mlp(x))
        return x


class AttentiveProbe(nn.Module):
    """
    Generic attentive probe matching V-JEPA 2.1 paper architecture.

    Args:
        encoder_dim:  dimension of frozen encoder output tokens (1408 for ViT-G)
        probe_dim:    internal dimension of probe transformer
        num_heads:    attention heads (default 16)
        num_blocks:   total transformer blocks (default 4)
        num_queries:  number of learnable query tokens (1 for classification, 3 for anticipation)
        num_classes:  output classes per query
    """

    def __init__(
        self,
        encoder_dim: int = 1408,
        probe_dim: int = 1408,
        num_heads: int = 16,
        num_blocks: int = 4,
        num_queries: int = 1,
        num_classes: int = 400,
    ):
        super().__init__()
        self.num_queries = num_queries

        # Project encoder tokens to probe_dim if needed
        self.input_proj = nn.Identity() if encoder_dim == probe_dim else nn.Linear(encoder_dim, probe_dim)

        # Blocks 0..num_blocks-2: self-attention; last block: cross-attention
        self.blocks = nn.ModuleList([
            TransformerBlock(probe_dim, num_heads, cross_attention=(i == num_blocks - 1))
            for i in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(probe_dim)

        # Learnable query tokens (one per output head)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, probe_dim))
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # Per-query classifier
        self.classifiers = nn.ModuleList([
            nn.Linear(probe_dim, num_classes) for _ in range(num_queries)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N, D) encoder output tokens (possibly from multiple layers concatenated)
        Returns:
            logits: (B, num_queries, num_classes) if num_queries > 1
                    (B, num_classes) if num_queries == 1
        """
        B = tokens.shape[0]
        x = self.input_proj(tokens)  # (B, N, probe_dim)

        # Self-attention blocks
        for block in self.blocks[:-1]:
            x = block(x)

        # Cross-attention: queries attend to encoder tokens
        queries = self.query_tokens.expand(B, -1, -1)  # (B, Q, D)
        queries = self.norm(self.blocks[-1](queries, context=x))  # (B, Q, D)

        # Residual add back (as described in paper: "output added back to query token as residual")
        queries = queries + self.query_tokens.expand(B, -1, -1)

        logits = torch.stack([self.classifiers[i](queries[:, i]) for i in range(self.num_queries)], dim=1)
        # (B, Q, num_classes)

        if self.num_queries == 1:
            return logits.squeeze(1)  # (B, num_classes)
        return logits  # (B, Q, num_classes)


# ---------------------------------------------------------------------------
# Action Anticipation Probe (EK100 / Ego4D)
# ---------------------------------------------------------------------------

class ActionAnticipationProbe(nn.Module):
    """
    Action anticipation probe for EK100.
    Predicts verb (97 classes), noun (300 classes), and action jointly.
    Following V-JEPA 2.1 Appendix C.6: 3 query tokens, one per classifier.
    """

    VERB_CLASSES = 97
    NOUN_CLASSES = 300
    ACTION_CLASSES = 3568

    def __init__(self, encoder_dim: int = 1408, probe_dim: int = 1408, num_heads: int = 16):
        super().__init__()
        self.verb_probe = AttentiveProbe(encoder_dim, probe_dim, num_heads, num_queries=1, num_classes=self.VERB_CLASSES)
        self.noun_probe = AttentiveProbe(encoder_dim, probe_dim, num_heads, num_queries=1, num_classes=self.NOUN_CLASSES)
        # Action = joint (verb, noun) — treated as top-k over noun-verb product
        # In practice we train separate heads and rank by product probability
        # (following the EK100 evaluation protocol)

    def forward(self, tokens: torch.Tensor):
        verb_logits = self.verb_probe(tokens)  # (B, V)
        noun_logits = self.noun_probe(tokens)  # (B, N)
        return verb_logits, noun_logits


# ---------------------------------------------------------------------------
# Focal Loss (used by V-JEPA 2.1 for action anticipation training)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal
