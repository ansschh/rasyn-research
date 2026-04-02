"""
Forward NMR Model — Structure → Chemical Shifts.

Predicts per-atom NMR chemical shifts from a molecular graph.
Outputs heteroscedastic predictions (mean + uncertainty per atom).

This is Component A from the proposal.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ForwardNMRModel(nn.Module):
    """
    Forward NMR predictor: molecular graph → per-atom chemical shifts.

    For each atom, predicts:
    - shift_mean: predicted chemical shift (ppm)
    - shift_logvar: log-variance (heteroscedastic uncertainty)

    Only predicts for atoms of the target nucleus (C for 13C, H for 1H).
    """

    def __init__(
        self,
        atom_repr_dim: int = 256,
        hidden_dims: list[int] = [256, 128],
        dropout: float = 0.1,
    ):
        super().__init__()

        # Nucleus-specific heads
        layers = []
        in_dim = atom_repr_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)])
            in_dim = h

        self.trunk = nn.Sequential(*layers)

        # Predict mean and log-variance
        self.shift_mean_head = nn.Linear(in_dim, 1)
        self.shift_logvar_head = nn.Linear(in_dim, 1)

    def forward(
        self,
        atom_repr: torch.Tensor,          # (n_atoms, atom_repr_dim)
        atom_mask: Optional[torch.Tensor] = None,  # (n_atoms,) 1=target nucleus
    ) -> dict[str, torch.Tensor]:
        """Predict chemical shifts for target atoms."""
        h = self.trunk(atom_repr)

        shift_mean = self.shift_mean_head(h).squeeze(-1)
        shift_logvar = self.shift_logvar_head(h).squeeze(-1)

        return {
            "shift_mean": shift_mean,
            "shift_logvar": shift_logvar,
            "shift_std": torch.exp(0.5 * shift_logvar),
        }

    def compute_loss(
        self,
        predictions: dict,
        true_shifts: torch.Tensor,          # (n_atoms,) ground truth shifts
        atom_mask: torch.Tensor,            # (n_atoms,) 1=has shift label
    ) -> dict[str, torch.Tensor]:
        """
        Heteroscedastic Gaussian NLL loss for shift prediction.
        """
        mean = predictions["shift_mean"]
        logvar = predictions["shift_logvar"]

        # NLL = 0.5 * (logvar + (y - mean)^2 / exp(logvar))
        nll = 0.5 * (logvar + (true_shifts - mean) ** 2 * torch.exp(-logvar))
        nll = (nll * atom_mask).sum() / atom_mask.sum().clamp(min=1)

        # Also compute MAE for monitoring
        with torch.no_grad():
            mae = ((mean - true_shifts).abs() * atom_mask).sum() / atom_mask.sum().clamp(min=1)

        return {
            "total": nll,
            "nll": nll,
            "mae": mae,
        }

    @torch.no_grad()
    def predict_spectrum(
        self,
        atom_repr: torch.Tensor,
        atom_mask: torch.Tensor,
        spectrum_range: tuple[float, float] = (0.0, 220.0),
        n_bins: int = 1000,
        broadening: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate a continuous spectrum from predicted shifts.
        Each peak is a Gaussian centered at the predicted shift.

        Returns: (n_bins,) spectrum intensities
        """
        preds = self.forward(atom_repr, atom_mask)
        shifts = preds["shift_mean"][atom_mask > 0]
        stds = preds["shift_std"][atom_mask > 0]

        # Create bin positions
        bins = torch.linspace(spectrum_range[0], spectrum_range[1], n_bins, device=shifts.device)

        # Construct spectrum as sum of Gaussians
        spectrum = torch.zeros(n_bins, device=shifts.device)
        for shift, std in zip(shifts, stds):
            width = max(std.item(), broadening)
            peak = torch.exp(-0.5 * ((bins - shift) / width) ** 2) / (width * 2.507)
            spectrum = spectrum + peak

        return spectrum


def optimal_transport_spectral_loss(
    pred_peaks: torch.Tensor,   # (n_pred, 2) [position, intensity]
    true_peaks: torch.Tensor,   # (n_true, 2) [position, intensity]
    reg: float = 0.1,
) -> torch.Tensor:
    """
    Compute Sinkhorn OT distance between predicted and true spectral peaks.

    More physically meaningful than bin-wise MSE:
    - handles small peak shifts gracefully
    - handles missing/extra peaks
    - intensity-weighted alignment
    """
    if len(pred_peaks) == 0 or len(true_peaks) == 0:
        return torch.tensor(0.0, device=pred_peaks.device if len(pred_peaks) > 0 else true_peaks.device)

    # Cost matrix: pairwise distance in peak space
    # Position difference is main cost, intensity mismatch is secondary
    pos_cost = (pred_peaks[:, 0:1] - true_peaks[:, 0:1].T) ** 2
    int_cost = (pred_peaks[:, 1:2] - true_peaks[:, 1:2].T) ** 2 * 0.1
    C = pos_cost + int_cost

    # Sinkhorn iterations
    n, m = C.shape
    a = torch.ones(n, device=C.device) / n  # uniform source weights
    b = torch.ones(m, device=C.device) / m  # uniform target weights

    K = torch.exp(-C / reg)

    u = torch.ones(n, device=C.device)
    for _ in range(50):  # Sinkhorn iterations
        v = b / (K.T @ u + 1e-8)
        u = a / (K @ v + 1e-8)

    transport = torch.diag(u) @ K @ torch.diag(v)
    ot_loss = (transport * C).sum()

    return ot_loss
