"""
Peak Set Encoder — Set Transformer for spectral peak sets.

Treats spectra NOT as sequences or images, but as UNORDERED SETS of peaks.
Each peak is a feature vector: [position, intensity, modality, uncertainty].

Uses Induced Set Attention Blocks (ISAB) for O(n*m) complexity
instead of O(n^2) full attention.

This is the core representational novelty from the proposal (H1).
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    """Standard multihead attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, _ = query.shape

        Q = self.W_q(query).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(out)


class ISAB(nn.Module):
    """
    Induced Set Attention Block.

    Uses m inducing points to reduce complexity from O(n^2) to O(n*m).
    Key building block of the Set Transformer.
    """

    def __init__(self, d_model: int, n_heads: int, n_inducing: int = 32, dropout: float = 0.1):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, n_inducing, d_model) * 0.02)

        self.attn1 = MultiheadAttention(d_model, n_heads, dropout)
        self.attn2 = MultiheadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))
        self.ff2 = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        I = self.inducing_points.expand(B, -1, -1)

        # Step 1: inducing points attend to input
        H = self.norm1(I + self.attn1(I, x, x, mask))
        H = H + self.ff1(H)

        # Step 2: input attends to inducing points
        out = self.norm2(x + self.attn2(x, H, H))
        out = out + self.ff2(out)

        return out


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.
    Reduces set to fixed number of seed vectors.
    """

    def __init__(self, d_model: int, n_heads: int, n_seeds: int = 1, dropout: float = 0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.attn = MultiheadAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        S = self.seeds.expand(B, -1, -1)
        out = self.norm(S + self.attn(S, x, x, mask))
        return out


class PeakSetEncoder(nn.Module):
    """
    Set Transformer encoder for spectral peak sets.

    Input: variable-length set of peaks, each with features
           [position, intensity, modality_id, uncertainty]

    Output: fixed-size spectral embedding

    Permutation-invariant by construction — no positional encoding needed.
    This is the key difference from sequence-based spectral encoders.
    """

    def __init__(
        self,
        peak_feature_dim: int = 4,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 3,
        n_inducing: int = 32,
        max_peaks: int = 200,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Project peak features to d_model
        self.peak_proj = nn.Sequential(
            nn.Linear(peak_feature_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Modality embedding (0=1H, 1=13C, 2=MS/MS, etc.)
        self.modality_embedding = nn.Embedding(8, d_model)

        # Stack of ISABs
        self.isab_layers = nn.ModuleList([
            ISAB(d_model, n_heads, n_inducing, dropout)
            for _ in range(n_layers)
        ])

        # Pooling to fixed-size output
        self.pool = PMA(d_model, n_heads, n_seeds=1, dropout=dropout)

        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        peak_features: torch.Tensor,         # (batch, max_peaks, peak_feature_dim)
        peak_mask: Optional[torch.Tensor] = None,  # (batch, max_peaks) True=padding
    ) -> dict[str, torch.Tensor]:
        """
        Encode a set of spectral peaks.

        Returns:
            peak_repr: (batch, max_peaks, d_model) per-peak representations
            spectral_repr: (batch, d_model) pooled spectral embedding
        """
        # Project peak features
        h = self.peak_proj(peak_features)

        # Add modality embedding
        modality_idx = peak_features[:, :, 2].long().clamp(0, 7)
        h = h + self.modality_embedding(modality_idx)

        # Apply ISAB layers
        for isab in self.isab_layers:
            h = isab(h, mask=peak_mask)

        peak_repr = h

        # Pool to single vector
        spectral_repr = self.pool(h, mask=peak_mask)
        spectral_repr = self.output_norm(spectral_repr.squeeze(1))

        return {
            "peak_repr": peak_repr,
            "spectral_repr": spectral_repr,
        }
