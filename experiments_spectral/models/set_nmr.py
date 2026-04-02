"""
SetNMR — Set-Valued NMR Chemical Shift Prediction with Hungarian Matching.

Instead of predicting shift-per-atom (which requires reliable atom-level labels),
predict the SET of shifts a molecule produces and use optimal assignment for the loss.

Key insight: nmrshiftdb2 has ~22% misassigned atom indices (e.g., 13C shifts indexed
to O atoms instead of C). Per-atom prediction can never learn clean signal from this.
Set prediction sidesteps the problem entirely.

Architecture:
  MPNN → filter to target atoms (C for 13C) → predict shift per atom →
  Hungarian matching loss finds optimal assignment to true peaks

This is analogous to DETR's set prediction for object detection.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment


class SetNMRPredictor(nn.Module):
    """
    Set-valued NMR shift predictor.

    Given per-atom representations from an MPNN, predicts a chemical shift
    for each atom of the target element. The loss uses Hungarian matching
    to find the optimal assignment between predicted and true shifts.
    """

    def __init__(
        self,
        atom_repr_dim: int = 256,
        hidden_dims: list[int] = [256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()

        layers = []
        in_dim = atom_repr_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.LayerNorm(h), nn.Dropout(dropout)])
            in_dim = h

        self.trunk = nn.Sequential(*layers)
        self.shift_mean = nn.Linear(in_dim, 1)
        self.shift_logvar = nn.Linear(in_dim, 1)

    def forward(self, atom_repr: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Predict shifts for all input atoms.

        Args:
            atom_repr: (n_target_atoms, atom_repr_dim) representations of target-element atoms

        Returns:
            shift_mean: (n_target_atoms,) predicted shifts
            shift_logvar: (n_target_atoms,) log-variance (heteroscedastic uncertainty)
        """
        h = self.trunk(atom_repr)
        mean = self.shift_mean(h).squeeze(-1)
        logvar = self.shift_logvar(h).squeeze(-1)

        return {
            "shift_mean": mean,
            "shift_logvar": logvar,
            "shift_std": torch.exp(0.5 * logvar),
        }


def hungarian_shift_loss(
    pred_shifts: torch.Tensor,    # (n_pred,) predicted shifts
    true_shifts: torch.Tensor,    # (n_true,) ground truth shifts
    pred_logvar: Optional[torch.Tensor] = None,  # (n_pred,) for heteroscedastic NLL
) -> dict[str, torch.Tensor]:
    """
    Hungarian matching loss for set-valued shift prediction.

    Finds the optimal 1-to-1 assignment between predicted and true shifts
    using the Hungarian algorithm, then computes the loss on matched pairs.

    Handles unequal set sizes:
    - If n_pred > n_true: extra predictions are penalized (unmatched penalty)
    - If n_pred < n_true: missed peaks are penalized
    """
    n_pred = len(pred_shifts)
    n_true = len(true_shifts)

    if n_pred == 0 or n_true == 0:
        # Edge case: penalize for missing everything
        return {
            "total": torch.tensor(50.0, device=pred_shifts.device, requires_grad=True),
            "matched_mae": torch.tensor(50.0),
            "n_matched": 0,
        }

    # Build cost matrix: pairwise absolute differences
    # (n_pred, n_true)
    cost_matrix = (pred_shifts.unsqueeze(1) - true_shifts.unsqueeze(0)).abs()

    # Handle unequal sizes by padding with a penalty cost
    unmatched_penalty = 50.0  # ppm — penalty for unmatched peaks
    n_max = max(n_pred, n_true)

    if n_pred < n_max or n_true < n_max:
        padded_cost = torch.full((n_max, n_max), unmatched_penalty, device=cost_matrix.device)
        padded_cost[:n_pred, :n_true] = cost_matrix
        cost_np = padded_cost.detach().cpu().numpy()
    else:
        cost_np = cost_matrix.detach().cpu().numpy()

    # Solve assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_np)

    # Compute loss only on valid matches (not padded)
    total_loss = torch.tensor(0.0, device=pred_shifts.device, requires_grad=True)
    matched_errors = []

    for r, c in zip(row_ind, col_ind):
        if r < n_pred and c < n_true:
            # Real match
            if pred_logvar is not None:
                # Heteroscedastic NLL
                lv = pred_logvar[r]
                nll = 0.5 * (lv + (pred_shifts[r] - true_shifts[c]) ** 2 * torch.exp(-lv))
                total_loss = total_loss + nll
            else:
                # L1 loss
                total_loss = total_loss + (pred_shifts[r] - true_shifts[c]).abs()
            matched_errors.append((pred_shifts[r] - true_shifts[c]).abs().item())
        else:
            # Unmatched — add penalty
            total_loss = total_loss + unmatched_penalty

    n_matched = len(matched_errors)
    total_loss = total_loss / max(n_max, 1)

    matched_mae = np.mean(matched_errors) if matched_errors else 50.0

    return {
        "total": total_loss,
        "matched_mae": torch.tensor(matched_mae),
        "n_matched": n_matched,
    }


def spectrum_reconstruction_loss(
    pred_shifts: torch.Tensor,     # (n_pred,)
    true_shifts: torch.Tensor,     # (n_true,)
    spectrum_range: tuple[float, float] = (0.0, 220.0),
    n_bins: int = 500,
    broadening: float = 2.0,
) -> torch.Tensor:
    """
    Construct continuous spectra from both predicted and true shift sets
    as sum-of-Gaussians, then compute cosine similarity loss.

    This provides smooth gradients even when peaks are far apart,
    complementing the discrete Hungarian matching.
    """
    device = pred_shifts.device
    bins = torch.linspace(spectrum_range[0], spectrum_range[1], n_bins, device=device)

    # Construct predicted spectrum
    pred_spec = torch.zeros(n_bins, device=device)
    for s in pred_shifts:
        pred_spec = pred_spec + torch.exp(-0.5 * ((bins - s) / broadening) ** 2)

    # Construct true spectrum
    true_spec = torch.zeros(n_bins, device=device)
    for s in true_shifts:
        true_spec = true_spec + torch.exp(-0.5 * ((bins - s) / broadening) ** 2)

    # Cosine similarity loss (1 - cos_sim)
    cos_sim = F.cosine_similarity(pred_spec.unsqueeze(0), true_spec.unsqueeze(0))
    return 1.0 - cos_sim.squeeze()


def combined_set_nmr_loss(
    pred_shifts: torch.Tensor,
    true_shifts: torch.Tensor,
    pred_logvar: Optional[torch.Tensor] = None,
    hungarian_weight: float = 1.0,
    spectrum_weight: float = 0.5,
    spectrum_range: tuple[float, float] = (0.0, 220.0),
) -> dict[str, torch.Tensor]:
    """
    Combined loss: Hungarian matching + spectrum reconstruction.

    Hungarian gives precise peak-level gradients.
    Spectrum reconstruction gives smooth long-range gradients.
    """
    # Hungarian matching
    hung = hungarian_shift_loss(pred_shifts, true_shifts, pred_logvar)

    # Spectrum reconstruction
    spec_loss = spectrum_reconstruction_loss(pred_shifts, true_shifts, spectrum_range=spectrum_range)

    total = hungarian_weight * hung["total"] + spectrum_weight * spec_loss

    return {
        "total": total,
        "hungarian_loss": hung["total"],
        "spectrum_loss": spec_loss,
        "matched_mae": hung["matched_mae"],
        "n_matched": hung["n_matched"],
    }
