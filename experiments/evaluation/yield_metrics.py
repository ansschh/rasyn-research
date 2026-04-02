"""
Evaluation metrics for yield-feasibility field models (Experiment 3).

Standard:
- RMSE, MAE, rank correlation for yield
- Brier score, AUROC, AUPRC for feasibility
- Expected Calibration Error (ECE)

Novel:
- Robustness-adjusted utility = yield * plateau_width
- OOD performance degradation
- Field smoothness / sensitivity measures
"""

import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score, brier_score_loss,
)
from scipy.stats import spearmanr


def yield_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Standard regression metrics for yield prediction."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_t, y_p = y_true[mask], y_pred[mask]

    if len(y_t) < 2:
        return {}

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_t, y_p))),
        "mae": float(mean_absolute_error(y_t, y_p)),
        "spearman_r": float(spearmanr(y_t, y_p).correlation),
        "r2": float(1 - np.sum((y_t - y_p) ** 2) / np.sum((y_t - y_t.mean()) ** 2)),
    }


def feasibility_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray) -> dict[str, float]:
    """Classification metrics for feasibility prediction."""
    if len(np.unique(y_true)) < 2:
        return {"warning": "single_class"}

    return {
        "brier": float(brier_score_loss(y_true, y_pred_prob)),
        "auroc": float(roc_auc_score(y_true, y_pred_prob)),
        "auprc": float(average_precision_score(y_true, y_pred_prob)),
    }


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    task: str = "regression",
) -> float:
    """
    Expected Calibration Error.

    For regression: bin by predicted confidence interval width, check coverage.
    For classification: bin by predicted probability, check accuracy.
    """
    if task == "classification":
        # Standard ECE for binary classification
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
        return float(ece)

    elif task == "regression":
        # For regression with uncertainty: check if predicted intervals are calibrated
        # This requires y_pred_std; use residual-based approximation
        residuals = np.abs(y_true - y_pred)
        sorted_idx = np.argsort(residuals)
        n = len(residuals)
        # Ideal: residual quantiles should match prediction order
        ece = np.abs(np.arange(1, n + 1) / n - np.sort(y_pred[sorted_idx])).mean()
        return float(ece)

    return 0.0


# ─── Novel: Robustness-aware metrics ─────────────────────────────────────────

def robustness_adjusted_utility(
    yield_values: np.ndarray,
    plateau_widths: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Robustness-adjusted utility: balance yield against operating window size.

    U = yield^(1-α) * plateau_width^α

    A reaction with yield=0.7 and plateau_width=0.9 may be more useful than
    one with yield=0.9 and plateau_width=0.2.
    """
    return (yield_values ** (1 - alpha)) * (plateau_widths ** alpha)


def plateau_vs_peak_ranking(
    yields: np.ndarray,
    plateau_widths: np.ndarray,
    true_success_under_perturbation: np.ndarray,
) -> dict[str, float]:
    """
    Does the model correctly prefer broad plateaus over narrow peaks
    when the plateau actually survives perturbation better?

    Compares:
    - yield-only ranking
    - robustness-adjusted ranking
    against ground truth perturbation outcomes.
    """
    # Rank by yield only
    yield_ranking = np.argsort(-yields)

    # Rank by robustness-adjusted utility
    robust_utility = robustness_adjusted_utility(yields, plateau_widths)
    robust_ranking = np.argsort(-robust_utility)

    # Ground truth: rank by actual success under perturbation
    true_ranking = np.argsort(-true_success_under_perturbation)

    # Compute rank correlation with ground truth
    from scipy.stats import kendalltau

    yield_tau = kendalltau(yield_ranking, true_ranking).correlation
    robust_tau = kendalltau(robust_ranking, true_ranking).correlation

    return {
        "yield_only_kendall_tau": float(yield_tau) if not np.isnan(yield_tau) else 0.0,
        "robust_utility_kendall_tau": float(robust_tau) if not np.isnan(robust_tau) else 0.0,
        "improvement": float(robust_tau - yield_tau) if not np.isnan(robust_tau - yield_tau) else 0.0,
    }


def uncertainty_quality(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    epistemic_std: np.ndarray = None,
) -> dict[str, float]:
    """
    Evaluate quality of uncertainty estimates.

    - Negative log-likelihood under predicted Gaussian
    - Calibration of prediction intervals
    - Correlation between uncertainty and actual error
    """
    residuals = np.abs(y_true - y_pred_mean)
    std_safe = np.maximum(y_pred_std, 1e-6)

    # NLL
    nll = 0.5 * np.log(2 * np.pi * std_safe ** 2) + 0.5 * (residuals / std_safe) ** 2
    mean_nll = float(nll.mean())

    # Calibration: fraction of true values within ±k*sigma
    within_1sig = (residuals < 1.0 * std_safe).mean()  # should be ~0.68
    within_2sig = (residuals < 2.0 * std_safe).mean()  # should be ~0.95

    # Uncertainty-error correlation
    corr = float(spearmanr(std_safe, residuals).correlation) if len(residuals) > 10 else 0.0

    results = {
        "nll": mean_nll,
        "within_1sigma": float(within_1sig),
        "within_2sigma": float(within_2sig),
        "uncertainty_error_corr": corr,
    }

    if epistemic_std is not None:
        # Epistemic uncertainty should be higher where errors are larger
        epi_corr = float(spearmanr(epistemic_std, residuals).correlation) if len(residuals) > 10 else 0.0
        results["epistemic_error_corr"] = epi_corr

    return results


# ─── Combined evaluation ─────────────────────────────────────────────────────

def evaluate_yield_field(
    y_true: np.ndarray,
    success_true: np.ndarray,
    predictions: dict,
) -> dict[str, float]:
    """Full evaluation suite for yield field model."""
    results = {}

    # Standard yield metrics
    if "yield_mean" in predictions:
        results.update(yield_regression_metrics(y_true, predictions["yield_mean"]))

    # Feasibility metrics
    if "feasibility_prob" in predictions:
        feas = feasibility_metrics(success_true, predictions["feasibility_prob"])
        results.update({f"feas_{k}": v for k, v in feas.items()})

    # Calibration
    if "feasibility_prob" in predictions:
        results["feas_ece"] = expected_calibration_error(
            success_true, predictions["feasibility_prob"], task="classification"
        )

    # Uncertainty quality
    if "yield_mean" in predictions and "yield_std" in predictions:
        uncert = uncertainty_quality(
            y_true, predictions["yield_mean"], predictions["yield_std"],
            predictions.get("yield_epistemic_std"),
        )
        results.update({f"uncert_{k}": v for k, v in uncert.items()})

    # Robustness metrics
    if "plateau_width" in predictions:
        results["mean_plateau_width"] = float(predictions["plateau_width"].mean())
        results["mean_robust_utility"] = float(
            robustness_adjusted_utility(
                predictions["yield_mean"], predictions["plateau_width"]
            ).mean()
        )

    return results
