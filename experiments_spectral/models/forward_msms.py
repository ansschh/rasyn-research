"""
Forward MS/MS Model — Structure → Fragmentation Spectrum.

Predicts mass spectrum peaks from a molecular graph + collision energy.
This is Component B from the proposal.

At moderate scale, we use a simple approach:
- Encode molecule with MPNN
- Condition on collision energy
- Predict peak set (m/z, intensity pairs)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardMSMSModel(nn.Module):
    """
    Forward MS/MS predictor: molecular graph + collision energy → fragmentation peaks.

    Output: predicted peak intensities on a binned m/z axis, plus
    a peak set representation.
    """

    def __init__(
        self,
        mol_repr_dim: int = 256,
        ce_embedding_dim: int = 32,
        hidden_dims: list[int] = [512, 256],
        max_mz: float = 1500.0,
        n_bins: int = 2000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_mz = max_mz
        self.n_bins = n_bins
        self.bin_width = max_mz / n_bins

        # Collision energy embedding
        self.ce_proj = nn.Sequential(
            nn.Linear(1, ce_embedding_dim),
            nn.GELU(),
            nn.Linear(ce_embedding_dim, ce_embedding_dim),
        )

        # Spectrum predictor
        layers = []
        in_dim = mol_repr_dim + ce_embedding_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout)])
            in_dim = h

        self.trunk = nn.Sequential(*layers)

        # Binned spectrum output
        self.spectrum_head = nn.Linear(in_dim, n_bins)

    def forward(
        self,
        mol_repr: torch.Tensor,        # (batch, mol_repr_dim)
        collision_energy: torch.Tensor,  # (batch, 1) normalized CE
    ) -> dict[str, torch.Tensor]:
        """
        Predict binned mass spectrum.

        Returns:
            spectrum: (batch, n_bins) predicted intensities (softmax normalized)
            log_spectrum: (batch, n_bins) log-intensities
        """
        ce_emb = self.ce_proj(collision_energy)
        h = torch.cat([mol_repr, ce_emb], dim=-1)
        h = self.trunk(h)

        logits = self.spectrum_head(h)
        spectrum = F.softmax(logits, dim=-1)

        return {
            "spectrum": spectrum,
            "log_spectrum": F.log_softmax(logits, dim=-1),
            "logits": logits,
        }

    def compute_loss(
        self,
        predictions: dict,
        true_spectrum: torch.Tensor,  # (batch, n_bins) ground truth binned spectrum
    ) -> dict[str, torch.Tensor]:
        """
        Compute loss between predicted and true spectra.

        Uses:
        - Cosine similarity loss (standard in MS/MS)
        - KL divergence (treats spectra as distributions)
        """
        pred = predictions["spectrum"]
        true = true_spectrum

        # Normalize true spectrum
        true_norm = true / true.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        pred_norm = pred

        # Cosine similarity
        cos_sim = F.cosine_similarity(pred_norm, true_norm, dim=-1).mean()
        cos_loss = 1.0 - cos_sim

        # KL divergence
        log_pred = predictions["log_spectrum"]
        kl = F.kl_div(log_pred, true_norm, reduction="batchmean", log_target=False)

        total = cos_loss + 0.1 * kl

        return {
            "total": total,
            "cosine_loss": cos_loss,
            "cosine_sim": cos_sim,
            "kl_div": kl,
        }

    def extract_peaks(
        self,
        spectrum: torch.Tensor,
        threshold: float = 0.01,
        max_peaks: int = 100,
    ) -> list[list[tuple[float, float]]]:
        """
        Extract discrete peaks from binned spectrum.
        Returns list of (mz, intensity) pairs per sample.
        """
        batch_peaks = []
        for spec in spectrum:
            # Find bins above threshold
            above = spec > threshold * spec.max()
            indices = torch.where(above)[0]

            if len(indices) == 0:
                batch_peaks.append([])
                continue

            peaks = []
            for idx in indices[:max_peaks]:
                mz = (idx.float() + 0.5) * self.bin_width
                intensity = spec[idx].item()
                peaks.append((mz.item(), intensity))

            # Sort by intensity (descending)
            peaks.sort(key=lambda x: x[1], reverse=True)
            batch_peaks.append(peaks[:max_peaks])

        return batch_peaks
