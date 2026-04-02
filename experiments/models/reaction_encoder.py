"""
Shared Reaction Encoder — Transformer over SMILES tokens.

This is the backbone shared by condition, yield, and feasibility models.
Small scale: d_model=256, 4 layers, ~2M params.

Supports:
- Encoding reaction SMILES into a fixed-size representation
- Masked token modeling for pretraining
- Fine-tuning into downstream heads
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ReactionEncoder(nn.Module):
    """
    Transformer encoder for reaction SMILES.

    Encodes a tokenized reaction SMILES into:
    - per-token representations (for masked LM pretraining)
    - a pooled reaction representation (for downstream tasks)
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        # Masked LM head for pretraining
        self.mlm_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq_len) token indices
            attention_mask: (batch, seq_len) 1=real, 0=padding

        Returns dict with:
            - hidden_states: (batch, seq_len, d_model) per-token representations
            - pooled: (batch, d_model) mean-pooled reaction representation
        """
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Create padding mask for transformer (True = ignore)
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None

        hidden = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        hidden = self.layer_norm(hidden)

        # Mean pooling over non-padded tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1)  # (batch, seq, 1)
            pooled = (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)

        return {
            "hidden_states": hidden,
            "pooled": pooled,
        }

    def mlm_forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        """Forward pass for masked language model pretraining."""
        out = self.forward(input_ids, attention_mask)
        logits = self.mlm_head(out["hidden_states"])  # (batch, seq, vocab)
        return logits
