"""
Evaluation metrics for condition prediction (Experiment 2).

Standard metrics:
- Top-k exact tuple match
- Marginal component match

Novel metrics (for the manifold model):
- Coverage@α: fraction of valid conditions inside predicted region
- Set calibration: does the predicted region match empirical viability?
- Condition window volume: size of predicted viable region (robustness proxy)
"""

from typing import Optional

import numpy as np
from sklearn.metrics import top_k_accuracy_score


# ─── Standard metrics ─────────────────────────────────────────────────────────

def topk_exact_match(pred_indices: np.ndarray, true_indices: np.ndarray, k: int = 1) -> float:
    """
    Top-k exact tuple match.
    A prediction is correct if the true tuple is in the top-k predictions.
    pred_indices: (n_samples, k, n_components) - top-k predicted condition tuples
    true_indices: (n_samples, n_components) - true condition tuples
    """
    n = len(true_indices)
    matches = 0
    for i in range(n):
        for j in range(min(k, pred_indices.shape[1])):
            if np.array_equal(pred_indices[i, j], true_indices[i]):
                matches += 1
                break
    return matches / n


def marginal_component_match(pred_indices: np.ndarray, true_indices: np.ndarray, k: int = 5) -> dict[str, float]:
    """
    Per-component top-k match.
    For each condition component (solvent, catalyst, etc.), what fraction of
    true values appear in the top-k predictions for that component?
    """
    n_components = true_indices.shape[1]
    results = {}

    for comp in range(n_components):
        correct = 0
        for i in range(len(true_indices)):
            true_val = true_indices[i, comp]
            pred_vals = pred_indices[i, :k, comp] if pred_indices.ndim == 3 else pred_indices[i, :k]
            if true_val in pred_vals:
                correct += 1
        results[f"component_{comp}_top{k}"] = correct / len(true_indices)

    results["mean_marginal"] = np.mean(list(results.values()))
    return results


# ─── Novel metrics for condition manifold ─────────────────────────────────────

def coverage_at_alpha(
    sampled_energies: np.ndarray,     # (n_reactions, n_samples) energy values from EBM
    true_condition_energy: np.ndarray, # (n_reactions,) energy of true condition
    alpha: float = 0.9,
) -> float:
    """
    Coverage@α: fraction of true conditions that fall within the
    α-level set of the predicted energy landscape.

    If the manifold model is well-calibrated, Coverage@0.9 should be ~0.9.

    Args:
        sampled_energies: energies of sampled conditions from the EBM
        true_condition_energy: energy of the actual observed condition
        alpha: quantile threshold
    """
    n = len(true_condition_energy)
    covered = 0

    for i in range(n):
        # Find energy threshold at the alpha quantile of sampled conditions
        threshold = np.percentile(sampled_energies[i], alpha * 100)
        # True condition is "covered" if its energy is below (or at) this threshold
        if true_condition_energy[i] <= threshold:
            covered += 1

    return covered / n


def set_calibration_error(
    predicted_probs: np.ndarray,  # (n_reactions, n_bins) predicted probability mass in each bin
    empirical_probs: np.ndarray,  # (n_reactions, n_bins) actual fraction of valid conditions in each bin
    n_bins: int = 10,
) -> float:
    """
    Set calibration error: how well does the predicted viable region
    match the empirical distribution of valid conditions?

    This extends standard calibration (for binary predictions) to
    set-valued predictions.

    Returns mean absolute calibration error across bins.
    """
    bin_errors = np.abs(predicted_probs - empirical_probs).mean(axis=0)
    return float(bin_errors.mean())


def condition_window_volume(
    energies_on_grid: np.ndarray,  # (n_reactions, grid_size) energies on condition grid
    threshold: float = 0.0,        # energy threshold for viability
) -> np.ndarray:
    """
    Estimate the volume of the viable condition region.

    For each reaction, count the fraction of grid points with energy below threshold.
    Larger volume = wider operating window = more robust.

    Returns: (n_reactions,) fraction of grid in viable region
    """
    viable = (energies_on_grid < threshold).astype(float)
    return viable.mean(axis=1)


def energy_gap_quality(
    pos_energies: np.ndarray,   # energy of known-good conditions
    neg_energies: np.ndarray,   # energy of random/bad conditions
) -> dict[str, float]:
    """
    How well does the energy function separate viable from non-viable conditions?
    """
    gap = neg_energies.mean() - pos_energies.mean()
    # AUROC-like: fraction of (pos, neg) pairs where pos has lower energy
    n_correct = 0
    n_total = 0
    for pe in pos_energies.ravel()[:1000]:  # subsample for speed
        for ne in neg_energies.ravel()[:1000]:
            n_total += 1
            if pe < ne:
                n_correct += 1
            elif pe == ne:
                n_correct += 0.5

    return {
        "energy_gap": float(gap),
        "separation_auroc": n_correct / max(n_total, 1),
    }


# ─── Combined evaluation ─────────────────────────────────────────────────────

def evaluate_condition_model(
    model_type: str,
    predictions: dict,
    targets: dict,
    coverage_thresholds: list[float] = [0.5, 0.7, 0.9, 0.95],
) -> dict[str, float]:
    """
    Run full evaluation suite for a condition prediction model.

    Args:
        model_type: "point" or "ebm"
        predictions: model outputs
        targets: ground truth

    Returns: dict of all metrics
    """
    results = {}

    if model_type == "point":
        # Standard metrics for point predictors
        for k in [1, 3, 5, 10]:
            if "pred_indices" in predictions and "true_indices" in targets:
                results[f"exact_match_top{k}"] = topk_exact_match(
                    predictions["pred_indices"], targets["true_indices"], k=k
                )

        if "pred_indices" in predictions and "true_indices" in targets:
            marginal = marginal_component_match(
                predictions["pred_indices"], targets["true_indices"], k=5
            )
            results.update(marginal)

    elif model_type == "ebm":
        # Manifold model metrics
        if "sampled_energies" in predictions and "true_energy" in predictions:
            for alpha in coverage_thresholds:
                results[f"coverage@{alpha}"] = coverage_at_alpha(
                    predictions["sampled_energies"],
                    predictions["true_energy"],
                    alpha=alpha,
                )

        if "grid_energies" in predictions:
            volumes = condition_window_volume(predictions["grid_energies"])
            results["mean_window_volume"] = float(volumes.mean())
            results["std_window_volume"] = float(volumes.std())

        if "pos_energies" in predictions and "neg_energies" in predictions:
            gap = energy_gap_quality(predictions["pos_energies"], predictions["neg_energies"])
            results.update(gap)

    return results
