"""
Boundary-Focused Active Learning — Acquisition functions for viability boundary estimation.

Standard active learning queries where uncertainty is highest (anywhere).
Our approach queries where the VIABILITY BOUNDARY is uncertain — near the
transition between success and failure in condition space.

This is the core of Experiment 5.
"""

from typing import Optional

import torch
import torch.nn as nn
import numpy as np


class BoundaryAcquisition:
    """
    Acquisition functions for active learning of viability boundaries.

    Given a pool of unlabeled condition points and a trained yield/feasibility model,
    select the next batch of experiments to run.
    """

    def __init__(self, method: str = "boundary_gradient"):
        self.method = method

    def select(
        self,
        model: nn.Module,
        reaction_repr: torch.Tensor,        # (1, reaction_dim) — fixed reaction
        candidate_conditions: torch.Tensor,  # (n_candidates, condition_dim)
        budget: int = 20,
    ) -> np.ndarray:
        """
        Select the next experiments from the candidate pool.

        Args:
            model: trained yield/feasibility field model (or ensemble)
            reaction_repr: encoded reaction (fixed for HTE-style experiments)
            candidate_conditions: pool of unlabeled condition points
            budget: number of experiments to select

        Returns:
            indices: (budget,) indices into candidate_conditions
        """
        scores = self._compute_scores(model, reaction_repr, candidate_conditions)
        # Select top-scoring candidates
        indices = np.argsort(scores)[-budget:]
        return indices

    def _compute_scores(
        self,
        model: nn.Module,
        reaction_repr: torch.Tensor,
        candidates: torch.Tensor,
    ) -> np.ndarray:
        """Compute acquisition scores for all candidates."""
        scorer = getattr(self, f"_score_{self.method}")
        return scorer(model, reaction_repr, candidates)

    def _score_random(
        self, model: nn.Module,
        reaction_repr: torch.Tensor,
        candidates: torch.Tensor,
    ) -> np.ndarray:
        """Random baseline — uniform scores."""
        return np.random.random(len(candidates))

    def _score_uncertainty(
        self,
        model: nn.Module,
        reaction_repr: torch.Tensor,
        candidates: torch.Tensor,
    ) -> np.ndarray:
        """Standard uncertainty sampling — max predictive entropy."""
        n = len(candidates)
        rxn_exp = reaction_repr.expand(n, -1)

        with torch.no_grad():
            preds = model(rxn_exp, candidates)

        # Epistemic uncertainty (ensemble disagreement)
        if "yield_epistemic_std" in preds:
            scores = preds["yield_epistemic_std"].cpu().numpy()
        elif "yield_std" in preds:
            scores = preds["yield_std"].cpu().numpy()
        else:
            # Fallback: binary entropy of feasibility
            p = preds["feasibility_prob"].cpu().numpy()
            p = np.clip(p, 1e-6, 1 - 1e-6)
            scores = -(p * np.log(p) + (1 - p) * np.log(1 - p))

        return scores

    def _score_expected_improvement(
        self,
        model: nn.Module,
        reaction_repr: torch.Tensor,
        candidates: torch.Tensor,
    ) -> np.ndarray:
        """Expected Improvement — standard BO acquisition."""
        from scipy.stats import norm

        n = len(candidates)
        rxn_exp = reaction_repr.expand(n, -1)

        with torch.no_grad():
            preds = model(rxn_exp, candidates)

        mean = preds["yield_mean"].cpu().numpy()
        std = preds["yield_std"].cpu().numpy() if "yield_std" in preds else np.ones_like(mean) * 0.1

        # Best observed so far (assume we know it)
        best_so_far = mean.max()  # approximate

        z = (mean - best_so_far) / np.maximum(std, 1e-6)
        ei = std * (z * norm.cdf(z) + norm.pdf(z))
        return ei

    def _score_boundary_gradient(
        self,
        model: nn.Module,
        reaction_repr: torch.Tensor,
        candidates: torch.Tensor,
    ) -> np.ndarray:
        """
        Boundary-focused acquisition: query where feasibility gradient is steepest.

        The idea: the viability boundary is where p_success transitions from
        high to low. Points near this boundary are the most informative for
        learning the shape of the viable region.

        Score = ||∇_c p_success(c)|| — gradient magnitude of feasibility w.r.t. conditions.
        """
        n = len(candidates)
        rxn_exp = reaction_repr.expand(n, -1).detach()
        cands = candidates.detach().requires_grad_(True)

        # Forward pass with gradients
        preds = model(rxn_exp, cands)
        feasibility = preds["feasibility_prob"]

        # Compute gradient of feasibility w.r.t. condition inputs
        grad = torch.autograd.grad(
            feasibility.sum(), cands,
            create_graph=False, retain_graph=False,
        )[0]

        # Score = gradient magnitude (high gradient = near boundary)
        scores = grad.norm(dim=-1).cpu().numpy()
        return scores

    def _score_boundary_uncertainty(
        self,
        model: nn.Module,
        reaction_repr: torch.Tensor,
        candidates: torch.Tensor,
    ) -> np.ndarray:
        """
        Boundary-focused acquisition (variant): query where feasibility is near 0.5
        AND uncertainty is high.

        Points where the model thinks p_success ≈ 0.5 are likely near the boundary.
        Among those, prefer points with high epistemic uncertainty.
        """
        n = len(candidates)
        rxn_exp = reaction_repr.expand(n, -1)

        with torch.no_grad():
            preds = model(rxn_exp, candidates)

        p = preds["feasibility_prob"].cpu().numpy()

        # Proximity to boundary (0.5)
        boundary_proximity = 1.0 - np.abs(2 * p - 1)  # max at p=0.5

        # Uncertainty
        if "feasibility_epistemic_std" in preds:
            uncertainty = preds["feasibility_epistemic_std"].cpu().numpy()
        elif "yield_epistemic_std" in preds:
            uncertainty = preds["yield_epistemic_std"].cpu().numpy()
        else:
            uncertainty = np.ones_like(p)

        # Combined: boundary proximity * uncertainty
        scores = boundary_proximity * (1.0 + uncertainty)
        return scores


# ─── Active Learning Loop ────────────────────────────────────────────────────

def run_active_learning_loop(
    model_class,
    model_kwargs: dict,
    train_fn,
    reaction_repr: torch.Tensor,
    all_conditions: torch.Tensor,      # (N, condition_dim) — full pool
    all_yields: torch.Tensor,          # (N,) — ground truth (oracle)
    all_success: torch.Tensor,         # (N,) — ground truth
    initial_fraction: float = 0.1,
    budget_per_round: int = 20,
    n_rounds: int = 20,
    acquisition_method: str = "boundary_gradient",
    seed: int = 42,
) -> dict:
    """
    Run a full active learning loop.

    Args:
        model_class: class to instantiate (e.g., YieldFieldEnsemble)
        model_kwargs: kwargs for model_class
        train_fn: callable(model, X_train, y_train) that trains the model
        reaction_repr: fixed reaction encoding
        all_conditions: full condition pool (ground truth available for all)
        all_yields: ground truth yields
        all_success: ground truth success labels
        initial_fraction: fraction of data to start with
        budget_per_round: experiments per round
        n_rounds: number of acquisition rounds
        acquisition_method: which acquisition function to use
        seed: random seed

    Returns:
        history: dict with metrics at each round
    """
    rng = np.random.RandomState(seed)
    N = len(all_conditions)
    n_initial = int(N * initial_fraction)

    # Initial random selection
    all_indices = np.arange(N)
    rng.shuffle(all_indices)
    labeled_indices = set(all_indices[:n_initial].tolist())
    unlabeled_indices = set(all_indices[n_initial:].tolist())

    acquirer = BoundaryAcquisition(method=acquisition_method)

    history = {
        "round": [],
        "n_labeled": [],
        "yield_rmse": [],
        "feasibility_auroc": [],
        "viability_iou": [],  # intersection over union of estimated vs true viable region
    }

    for round_idx in range(n_rounds + 1):
        # Train model on currently labeled data
        labeled_list = sorted(labeled_indices)
        X_train = all_conditions[labeled_list]
        y_train = all_yields[labeled_list]
        s_train = all_success[labeled_list]

        model = model_class(**model_kwargs)
        train_fn(model, reaction_repr, X_train, y_train, s_train)

        # Evaluate on ALL data
        with torch.no_grad():
            rxn_exp = reaction_repr.expand(N, -1)
            preds = model(rxn_exp, all_conditions)

        pred_yield = preds["yield_mean"].cpu().numpy()
        pred_feas = preds["feasibility_prob"].cpu().numpy()
        true_yield = all_yields.cpu().numpy()
        true_success = all_success.cpu().numpy()

        # Metrics
        rmse = np.sqrt(np.mean((pred_yield - true_yield) ** 2))

        from sklearn.metrics import roc_auc_score
        try:
            auroc = roc_auc_score(true_success, pred_feas)
        except ValueError:
            auroc = 0.5

        # Viability region IoU
        pred_viable = pred_feas > 0.5
        true_viable = true_success > 0.5
        intersection = (pred_viable & true_viable).sum()
        union = (pred_viable | true_viable).sum()
        iou = intersection / max(union, 1)

        history["round"].append(round_idx)
        history["n_labeled"].append(len(labeled_indices))
        history["yield_rmse"].append(float(rmse))
        history["feasibility_auroc"].append(float(auroc))
        history["viability_iou"].append(float(iou))

        # Acquire next batch (skip on last round)
        if round_idx < n_rounds and unlabeled_indices:
            unlabeled_list = sorted(unlabeled_indices)
            candidate_conds = all_conditions[unlabeled_list]

            selected_relative = acquirer.select(
                model, reaction_repr, candidate_conds, budget=budget_per_round
            )
            selected_absolute = [unlabeled_list[i] for i in selected_relative]

            for idx in selected_absolute:
                labeled_indices.add(idx)
                unlabeled_indices.discard(idx)

    return history
