"""
Baseline: Point Condition Predictor.

Predicts a single condition tuple (solvent, catalyst, reagent, base, temp, time)
for a given reaction. This is the standard approach in the literature.

Used as baseline for comparison with the condition manifold model.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointConditionPredictor(nn.Module):
    """
    Predict conditions as independent classification (discrete) and regression (continuous).

    This is deliberately simple — it's the baseline the manifold model must beat.
    """

    def __init__(
        self,
        reaction_dim: int = 256,
        discrete_vocab_sizes: dict[str, int] = None,
        n_continuous: int = 3,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        if discrete_vocab_sizes is None:
            discrete_vocab_sizes = {
                "solvent": 200,
                "catalyst": 200,
                "reagent": 300,
                "base": 100,
                "ligand": 100,
            }

        self.discrete_fields = list(discrete_vocab_sizes.keys())
        self.n_continuous = n_continuous

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(reaction_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Independent classification heads for discrete conditions
        self.discrete_heads = nn.ModuleDict({
            name: nn.Linear(hidden_dim, vocab_size)
            for name, vocab_size in discrete_vocab_sizes.items()
        })

        # Regression head for continuous conditions (temp, time, concentration)
        self.continuous_head = nn.Linear(hidden_dim, n_continuous)

    def forward(self, reaction_repr: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            reaction_repr: (batch, reaction_dim) from ReactionEncoder.pooled

        Returns:
            discrete_logits: dict of (batch, vocab_size) per discrete field
            continuous_pred: (batch, n_continuous) predicted continuous values
        """
        h = self.trunk(reaction_repr)

        discrete_logits = {
            name: head(h)
            for name, head in self.discrete_heads.items()
        }

        continuous_pred = self.continuous_head(h)

        return {
            "discrete_logits": discrete_logits,
            "continuous_pred": continuous_pred,
        }

    def compute_loss(
        self,
        predictions: dict,
        discrete_targets: torch.LongTensor,
        continuous_targets: torch.FloatTensor,
        continuous_mask: torch.FloatTensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute training loss.

        Args:
            predictions: output of forward()
            discrete_targets: (batch, n_discrete) target indices
            continuous_targets: (batch, n_continuous) target values (normalized)
            continuous_mask: (batch, n_continuous) 1 where target exists
        """
        total_loss = torch.tensor(0.0, device=discrete_targets.device)
        losses = {}

        # Discrete losses (cross-entropy per field)
        for i, name in enumerate(self.discrete_fields):
            logits = predictions["discrete_logits"][name]
            targets = discrete_targets[:, i]
            loss = F.cross_entropy(logits, targets, ignore_index=0)  # ignore PAD
            losses[f"ce_{name}"] = loss
            total_loss = total_loss + loss

        # Continuous loss (MSE where targets exist)
        if continuous_mask.sum() > 0:
            cont_pred = predictions["continuous_pred"]
            cont_loss = ((cont_pred - continuous_targets) ** 2 * continuous_mask).sum() / continuous_mask.sum().clamp(min=1)
            losses["mse_continuous"] = cont_loss
            total_loss = total_loss + cont_loss

        losses["total"] = total_loss
        return losses

    @torch.no_grad()
    def predict_topk(self, reaction_repr: torch.Tensor, k: int = 5) -> dict:
        """Predict top-k conditions for each discrete field."""
        preds = self.forward(reaction_repr)
        topk = {}
        for name, logits in preds["discrete_logits"].items():
            probs = F.softmax(logits, dim=-1)
            values, indices = probs.topk(k, dim=-1)
            topk[name] = {"indices": indices, "probs": values}
        topk["continuous"] = preds["continuous_pred"]
        return topk
