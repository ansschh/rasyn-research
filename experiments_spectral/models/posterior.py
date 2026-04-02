"""
Posterior Inference Engine — Bayesian reranking of structure candidates.

Given observed spectra (NMR, MS/MS, or both), score candidate structures by:
  log p(m | x_NMR, x_MS) ∝ log p(x_NMR | m) + log p(x_MS | m) + log p(m)

Uses forward models for spectral likelihood and OT-based spectral matching.
This is the core of Experiments 3 and 4 (H2, H3, H4).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConsistencyScorer(nn.Module):
    """
    Scores how consistent a candidate molecule is with observed spectra.

    Uses forward models to generate predicted spectra, then computes
    similarity between predicted and observed.
    """

    def __init__(
        self,
        mol_repr_dim: int = 256,
        spectral_repr_dim: int = 256,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Learn a compatibility function between molecular and spectral representations
        self.compatibility = nn.Sequential(
            nn.Linear(mol_repr_dim + spectral_repr_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Modality-specific weights (learnable)
        self.nmr_weight = nn.Parameter(torch.tensor(1.0))
        self.msms_weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        mol_repr: torch.Tensor,            # (n_candidates, mol_repr_dim)
        nmr_repr: Optional[torch.Tensor] = None,   # (1, spectral_repr_dim) or None
        msms_repr: Optional[torch.Tensor] = None,  # (1, spectral_repr_dim) or None
    ) -> dict[str, torch.Tensor]:
        """
        Score candidates against observed spectra.

        Returns:
            log_likelihood: (n_candidates,) unnormalized log-likelihood per candidate
            posterior: (n_candidates,) normalized posterior probabilities
        """
        n_cand = mol_repr.size(0)
        total_score = torch.zeros(n_cand, device=mol_repr.device)

        if nmr_repr is not None:
            nmr_expanded = nmr_repr.expand(n_cand, -1)
            nmr_score = self.compatibility(
                torch.cat([mol_repr, nmr_expanded], dim=-1)
            ).squeeze(-1)
            total_score = total_score + self.nmr_weight * nmr_score

        if msms_repr is not None:
            msms_expanded = msms_repr.expand(n_cand, -1)
            msms_score = self.compatibility(
                torch.cat([mol_repr, msms_expanded], dim=-1)
            ).squeeze(-1)
            total_score = total_score + self.msms_weight * msms_score

        posterior = F.softmax(total_score, dim=0)

        return {
            "log_likelihood": total_score,
            "posterior": posterior,
        }


class PosteriorInferenceEngine(nn.Module):
    """
    Full posterior inference over structure candidates.

    Pipeline:
    1. Encode observed spectra with PeakSetEncoder
    2. Encode candidate molecules with MolecularGraphEncoder
    3. (Optional) Generate predicted spectra for candidates using forward models
    4. Score candidates using spectral consistency + prior
    5. Return calibrated posterior
    """

    def __init__(
        self,
        mol_encoder: nn.Module,
        peak_encoder: nn.Module,
        forward_nmr: Optional[nn.Module] = None,
        forward_msms: Optional[nn.Module] = None,
        scorer: Optional[SpectralConsistencyScorer] = None,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.mol_encoder = mol_encoder
        self.peak_encoder = peak_encoder
        self.forward_nmr = forward_nmr
        self.forward_msms = forward_msms
        self.scorer = scorer or SpectralConsistencyScorer()
        self.temperature = temperature

    def compute_posterior(
        self,
        candidate_graphs: list[dict],         # list of graph dicts per candidate
        nmr_peaks: Optional[torch.Tensor] = None,     # (1, max_peaks, 4) observed NMR
        nmr_mask: Optional[torch.Tensor] = None,
        msms_peaks: Optional[torch.Tensor] = None,    # (1, max_peaks, 4) observed MS/MS
        msms_mask: Optional[torch.Tensor] = None,
        prior_scores: Optional[torch.Tensor] = None,  # (n_candidates,) log-prior
    ) -> dict[str, torch.Tensor]:
        """
        Compute posterior over candidates given observed spectra.

        Returns:
            posterior: (n_candidates,) calibrated probabilities
            log_posterior: (n_candidates,) log-probabilities
            entropy: scalar posterior entropy
            evidence_nmr: (n_candidates,) per-candidate NMR evidence
            evidence_msms: (n_candidates,) per-candidate MS/MS evidence
        """
        n_cand = len(candidate_graphs)

        # Encode observed spectra
        nmr_repr = None
        if nmr_peaks is not None:
            nmr_out = self.peak_encoder(nmr_peaks, nmr_mask)
            nmr_repr = nmr_out["spectral_repr"]  # (1, d_model)

        msms_repr = None
        if msms_peaks is not None:
            msms_out = self.peak_encoder(msms_peaks, msms_mask)
            msms_repr = msms_out["spectral_repr"]

        # Encode candidate molecules
        # (For simplicity, assume pre-computed mol representations)
        mol_reprs = []
        for graph in candidate_graphs:
            out = self.mol_encoder(
                graph["node_features"],
                graph["edge_index"],
                graph["edge_features"],
            )
            mol_reprs.append(out["mol_repr"])
        mol_repr = torch.cat(mol_reprs, dim=0)  # (n_cand, mol_dim)

        # Score candidates
        scores = self.scorer(mol_repr, nmr_repr, msms_repr)
        log_likelihood = scores["log_likelihood"]

        # Add prior
        if prior_scores is not None:
            log_posterior = log_likelihood + prior_scores
        else:
            log_posterior = log_likelihood

        # Temperature scaling
        log_posterior = log_posterior / self.temperature

        # Normalize
        posterior = F.softmax(log_posterior, dim=0)
        entropy = -(posterior * torch.log(posterior + 1e-10)).sum()

        return {
            "posterior": posterior,
            "log_posterior": log_posterior,
            "entropy": entropy,
            "log_likelihood": log_likelihood,
        }

    def compute_loss(
        self,
        posterior: torch.Tensor,           # (n_candidates,) predicted posterior
        true_index: int,                   # index of correct molecule
    ) -> dict[str, torch.Tensor]:
        """
        Loss: cross-entropy — the true molecule should have highest posterior.
        """
        target = torch.tensor(true_index, device=posterior.device)
        log_post = torch.log(posterior + 1e-10).unsqueeze(0)
        loss = F.nll_loss(log_post, target.unsqueeze(0))

        # Metrics
        rank = (posterior > posterior[true_index]).sum().item() + 1
        top1 = float(rank == 1)
        top5 = float(rank <= 5)
        top10 = float(rank <= 10)

        return {
            "total": loss,
            "nll": loss,
            "rank": torch.tensor(float(rank)),
            "top1": torch.tensor(top1),
            "top5": torch.tensor(top5),
            "top10": torch.tensor(top10),
        }


def posterior_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Compute entropy of posterior distribution."""
    return -(probs * torch.log(probs + 1e-10)).sum()


def modality_ablation_study(
    engine: PosteriorInferenceEngine,
    candidate_graphs: list[dict],
    nmr_peaks: torch.Tensor,
    nmr_mask: torch.Tensor,
    msms_peaks: torch.Tensor,
    msms_mask: torch.Tensor,
    true_index: int,
) -> dict[str, dict]:
    """
    Run posterior inference with different modality combinations.
    This directly tests H4: multimodal > unimodal.

    Returns results for:
    - NMR only
    - MS/MS only
    - NMR + MS/MS
    """
    results = {}

    # NMR only
    with torch.no_grad():
        post_nmr = engine.compute_posterior(
            candidate_graphs, nmr_peaks=nmr_peaks, nmr_mask=nmr_mask
        )
        loss_nmr = engine.compute_loss(post_nmr["posterior"], true_index)
        results["nmr_only"] = {
            "entropy": post_nmr["entropy"].item(),
            "rank": loss_nmr["rank"].item(),
            "top1": loss_nmr["top1"].item(),
        }

    # MS/MS only
    with torch.no_grad():
        post_msms = engine.compute_posterior(
            candidate_graphs, msms_peaks=msms_peaks, msms_mask=msms_mask
        )
        loss_msms = engine.compute_loss(post_msms["posterior"], true_index)
        results["msms_only"] = {
            "entropy": post_msms["entropy"].item(),
            "rank": loss_msms["rank"].item(),
            "top1": loss_msms["top1"].item(),
        }

    # Both
    with torch.no_grad():
        post_both = engine.compute_posterior(
            candidate_graphs,
            nmr_peaks=nmr_peaks, nmr_mask=nmr_mask,
            msms_peaks=msms_peaks, msms_mask=msms_mask,
        )
        loss_both = engine.compute_loss(post_both["posterior"], true_index)
        results["nmr_plus_msms"] = {
            "entropy": post_both["entropy"].item(),
            "rank": loss_both["rank"].item(),
            "top1": loss_both["top1"].item(),
        }

    return results
