"""
Evaluation metrics for boundary-focused active learning (Experiment 5).

Measures:
- How fast different acquisition strategies learn the viability region
- Quality of boundary estimation
- Data efficiency relative to random sampling
"""

import numpy as np
from typing import Optional


def viability_region_iou(
    predicted_viable: np.ndarray,   # (N,) boolean — model predicts viable
    true_viable: np.ndarray,        # (N,) boolean — ground truth
) -> float:
    """
    Intersection over Union of predicted vs true viability regions.
    Higher = better boundary estimation.
    """
    intersection = (predicted_viable & true_viable).sum()
    union = (predicted_viable | true_viable).sum()
    if union == 0:
        return 1.0  # both empty
    return float(intersection / union)


def viability_region_f1(
    predicted_viable: np.ndarray,
    true_viable: np.ndarray,
) -> dict[str, float]:
    """Precision, recall, F1 for viability region estimation."""
    tp = (predicted_viable & true_viable).sum()
    fp = (predicted_viable & ~true_viable).sum()
    fn = (~predicted_viable & true_viable).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def boundary_estimation_error(
    predicted_boundary: np.ndarray,  # (n_points, condition_dim) estimated boundary points
    true_boundary: np.ndarray,       # (n_points, condition_dim) true boundary points
) -> float:
    """
    Average distance from predicted boundary points to nearest true boundary point.
    Lower = better boundary estimation.
    """
    from scipy.spatial.distance import cdist
    if len(predicted_boundary) == 0 or len(true_boundary) == 0:
        return float("inf")

    dists = cdist(predicted_boundary, true_boundary)
    # Symmetric: average of predicted-to-true and true-to-predicted
    p_to_t = dists.min(axis=1).mean()
    t_to_p = dists.min(axis=0).mean()
    return float((p_to_t + t_to_p) / 2)


def data_efficiency_ratio(
    method_history: dict,  # from run_active_learning_loop
    baseline_history: dict,  # random baseline
    target_metric: str = "viability_iou",
    target_value: float = 0.8,
) -> float:
    """
    How many fewer experiments does the method need vs random baseline
    to reach a target metric value?

    Returns: ratio (< 1 means method is more efficient)
    """
    def _experiments_to_target(history, target):
        for i, val in enumerate(history[target_metric]):
            if val >= target:
                return history["n_labeled"][i]
        return history["n_labeled"][-1]  # never reached

    method_n = _experiments_to_target(method_history, target_value)
    baseline_n = _experiments_to_target(baseline_history, target_value)

    if baseline_n == 0:
        return 1.0
    return method_n / baseline_n


def learning_curve_auc(
    history: dict,
    metric: str = "viability_iou",
    normalize: bool = True,
) -> float:
    """
    Area under the learning curve (metric vs number of experiments).
    Higher = faster learning.
    """
    n_labeled = np.array(history["n_labeled"])
    values = np.array(history[metric])

    auc = np.trapz(values, n_labeled)

    if normalize and len(n_labeled) > 1:
        # Normalize by max possible area
        max_auc = (n_labeled[-1] - n_labeled[0]) * 1.0  # perfect = 1.0 everywhere
        auc = auc / max(max_auc, 1e-6)

    return float(auc)


def compare_acquisition_methods(
    all_histories: dict[str, dict],
    target_metrics: dict[str, float] = None,
) -> dict[str, dict[str, float]]:
    """
    Compare all acquisition methods on active learning efficiency.

    Returns a summary table suitable for the paper.
    """
    if target_metrics is None:
        target_metrics = {"viability_iou": 0.8}

    random_history = all_histories.get("random")

    summary = {}
    for method_name, history in all_histories.items():
        method_summary = {}

        # Final metrics
        method_summary["final_iou"] = history["viability_iou"][-1]
        method_summary["final_auroc"] = history["feasibility_auroc"][-1]
        method_summary["final_rmse"] = history["yield_rmse"][-1]

        # Learning curve AUC
        method_summary["learning_auc_iou"] = learning_curve_auc(history, "viability_iou")
        method_summary["learning_auc_auroc"] = learning_curve_auc(history, "feasibility_auroc")

        # Data efficiency vs random
        if random_history is not None:
            for metric_name, target_val in target_metrics.items():
                ratio = data_efficiency_ratio(history, random_history, metric_name, target_val)
                method_summary[f"efficiency_ratio_{metric_name}"] = ratio

        summary[method_name] = method_summary

    return summary
