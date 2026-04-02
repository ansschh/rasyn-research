"""
Evaluation metrics for spectral inference experiments.

Forward model metrics:
- Shift MAE/RMSE (per-atom)
- Spectral cosine similarity
- Optimal transport distance
- Calibration of shift uncertainty

Posterior inference metrics:
- Top-k recovery
- Posterior calibration
- Entropy reduction
- Modality ablation comparison

Measurement policy metrics:
- Average measurements to identify structure
- Entropy reduction efficiency
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ─── Forward model metrics ────────────────────────────────────────────────────

def shift_mae(pred_shifts: np.ndarray, true_shifts: np.ndarray, mask: np.ndarray = None) -> float:
    """Mean absolute error of predicted chemical shifts."""
    if mask is not None:
        pred_shifts = pred_shifts[mask > 0]
        true_shifts = true_shifts[mask > 0]
    return float(np.mean(np.abs(pred_shifts - true_shifts)))


def shift_rmse(pred_shifts: np.ndarray, true_shifts: np.ndarray, mask: np.ndarray = None) -> float:
    """Root mean squared error of predicted chemical shifts."""
    if mask is not None:
        pred_shifts = pred_shifts[mask > 0]
        true_shifts = true_shifts[mask > 0]
    return float(np.sqrt(np.mean((pred_shifts - true_shifts) ** 2)))


def spectral_cosine_similarity(pred_spectrum: np.ndarray, true_spectrum: np.ndarray) -> float:
    """Cosine similarity between two spectra (standard MS/MS metric)."""
    dot = np.dot(pred_spectrum, true_spectrum)
    norm = np.linalg.norm(pred_spectrum) * np.linalg.norm(true_spectrum)
    if norm < 1e-10:
        return 0.0
    return float(dot / norm)


def shift_calibration(
    pred_shifts: np.ndarray,
    pred_stds: np.ndarray,
    true_shifts: np.ndarray,
    mask: np.ndarray = None,
) -> dict[str, float]:
    """Evaluate calibration of shift uncertainty estimates."""
    if mask is not None:
        pred_shifts = pred_shifts[mask > 0]
        pred_stds = pred_stds[mask > 0]
        true_shifts = true_shifts[mask > 0]

    residuals = np.abs(true_shifts - pred_shifts)
    stds = np.maximum(pred_stds, 1e-6)

    within_1sig = (residuals < 1.0 * stds).mean()  # should be ~0.68
    within_2sig = (residuals < 2.0 * stds).mean()  # should be ~0.95

    # Correlation between uncertainty and error
    corr = spearmanr(stds, residuals).correlation if len(residuals) > 5 else 0.0

    return {
        "within_1sigma": float(within_1sig),
        "within_2sigma": float(within_2sig),
        "uncertainty_error_corr": float(corr) if not np.isnan(corr) else 0.0,
    }


# ─── Posterior inference metrics ──────────────────────────────────────────────

def topk_recovery(posterior: np.ndarray, true_index: int, k_values: list[int] = [1, 5, 10]) -> dict[str, float]:
    """Check if true molecule is in top-k of posterior."""
    sorted_indices = np.argsort(-posterior)
    rank = np.where(sorted_indices == true_index)[0][0] + 1

    results = {"rank": float(rank)}
    for k in k_values:
        results[f"top{k}"] = float(rank <= k)
    return results


def posterior_calibration(
    posteriors: list[np.ndarray],
    true_indices: list[int],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error for posterior predictions.
    The posterior probability assigned to the true molecule should be calibrated.
    """
    confidences = []
    correct = []

    for post, true_idx in zip(posteriors, true_indices):
        top_idx = np.argmax(post)
        confidences.append(post[top_idx])
        correct.append(float(top_idx == true_idx))

    confidences = np.array(confidences)
    correct = np.array(correct)

    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() / len(confidences) * abs(bin_acc - bin_conf)

    return float(ece)


def modality_comparison(results: dict[str, dict]) -> dict[str, float]:
    """
    Summarize modality ablation results.

    Input: {"nmr_only": {...}, "msms_only": {...}, "nmr_plus_msms": {...}}
    Output: improvement metrics
    """
    summary = {}
    for mode, metrics in results.items():
        summary[f"{mode}_rank"] = metrics.get("rank", float("inf"))
        summary[f"{mode}_top1"] = metrics.get("top1", 0.0)
        summary[f"{mode}_entropy"] = metrics.get("entropy", 0.0)

    # Improvement from combining modalities
    if "nmr_only" in results and "msms_only" in results and "nmr_plus_msms" in results:
        best_single = max(results["nmr_only"].get("top1", 0), results["msms_only"].get("top1", 0))
        combined = results["nmr_plus_msms"].get("top1", 0)
        summary["multimodal_improvement"] = combined - best_single

        nmr_entropy = results["nmr_only"].get("entropy", 0)
        combined_entropy = results["nmr_plus_msms"].get("entropy", 0)
        if nmr_entropy > 0:
            summary["entropy_reduction_pct"] = (nmr_entropy - combined_entropy) / nmr_entropy * 100

    return summary


# ─── Measurement policy metrics ──────────────────────────────────────────────

def measurements_to_target(
    histories: list[dict],
    target_metric: str = "top1",
    target_value: float = 1.0,
) -> float:
    """Average number of measurements needed to reach target."""
    counts = []
    for h in histories:
        values = h.get(target_metric, [])
        found = False
        for i, v in enumerate(values):
            if v >= target_value:
                counts.append(i + 1)
                found = True
                break
        if not found:
            counts.append(len(values) + 1)  # didn't reach target

    return float(np.mean(counts))


def policy_comparison(
    policy_results: dict[str, dict],
) -> dict[str, dict[str, float]]:
    """
    Compare measurement policies.

    Returns summary table for the paper.
    """
    summary = {}
    for policy_name, results in policy_results.items():
        summary[policy_name] = {
            "avg_final_rank": results.get("avg_final_rank", float("inf")),
            "avg_final_top1": results.get("avg_final_top1", 0.0),
            "avg_measurements_to_top1": results.get("avg_measurements_to_top1", float("inf")),
            "avg_entropy_reduction": results.get("avg_entropy_reduction", 0.0),
        }
    return summary
