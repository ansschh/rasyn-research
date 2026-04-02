"""
Molecular Graph Encoder — Message Passing Neural Network.

Encodes a molecular graph into:
- Per-atom representations (for shift prediction)
- Pooled molecular representation (for candidate scoring)

Small scale: ~2M params.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MPNNLayer(nn.Module):
    """Single message passing layer with edge features."""

    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.message_fn = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_fn = nn.GRUCell(hidden_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_feats: torch.Tensor,     # (n_atoms, node_dim)
        edge_index: torch.LongTensor,  # (2, n_edges)
        edge_feats: torch.Tensor,      # (n_edges, edge_dim)
    ) -> torch.Tensor:
        src, dst = edge_index
        n_nodes = node_feats.size(0)

        # Compute messages
        src_feats = node_feats[src]
        dst_feats = node_feats[dst]
        messages = self.message_fn(torch.cat([src_feats, dst_feats, edge_feats], dim=-1))

        # Aggregate (sum)
        agg = torch.zeros(n_nodes, messages.size(-1), device=node_feats.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)

        # Update
        updated = self.update_fn(agg, node_feats)
        return self.norm(updated)


class MolecularGraphEncoder(nn.Module):
    """
    MPNN encoder for molecular graphs.

    Input: molecular graph (node features, edge index, edge features)
    Output: per-atom representations + pooled molecular representation
    """

    def __init__(
        self,
        node_input_dim: int = 13,
        edge_input_dim: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project input features
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_input_dim, hidden_dim // 4)

        # Message passing layers
        self.mp_layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim // 4, hidden_dim)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        # Pooling: attention-weighted readout
        self.pool_attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.LongTensor,
        edge_features: torch.Tensor,
        batch_idx: Optional[torch.LongTensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            node_features: (total_atoms, node_input_dim)
            edge_index: (2, total_edges)
            edge_features: (total_edges, edge_input_dim)
            batch_idx: (total_atoms,) mapping atoms to molecules in batch

        Returns:
            atom_repr: (total_atoms, hidden_dim) per-atom representations
            mol_repr: (batch_size, hidden_dim) pooled molecular representations
        """
        h = self.node_proj(node_features)
        e = self.edge_proj(edge_features)

        for mp_layer in self.mp_layers:
            h = h + self.dropout(mp_layer(h, edge_index, e))

        atom_repr = h

        # Attention-weighted pooling
        attn_weights = self.pool_attn(h).squeeze(-1)  # (total_atoms,)

        if batch_idx is not None:
            # Softmax per molecule
            max_val = torch.zeros(batch_idx.max() + 1, device=h.device).scatter_reduce_(
                0, batch_idx, attn_weights, reduce="amax", include_self=False
            )
            attn_weights = attn_weights - max_val[batch_idx]
            attn_weights = torch.exp(attn_weights)
            attn_sum = torch.zeros(batch_idx.max() + 1, device=h.device).scatter_add_(
                0, batch_idx, attn_weights
            )
            attn_weights = attn_weights / attn_sum[batch_idx].clamp(min=1e-6)

            # Weighted sum
            weighted = h * attn_weights.unsqueeze(-1)
            mol_repr = torch.zeros(batch_idx.max() + 1, self.hidden_dim, device=h.device)
            mol_repr.scatter_add_(0, batch_idx.unsqueeze(-1).expand_as(weighted), weighted)
        else:
            attn_weights = F.softmax(attn_weights, dim=0)
            mol_repr = (h * attn_weights.unsqueeze(-1)).sum(dim=0, keepdim=True)

        return {
            "atom_repr": atom_repr,
            "mol_repr": mol_repr,
        }
