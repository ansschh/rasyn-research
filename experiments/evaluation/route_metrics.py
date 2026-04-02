"""
Evaluation metrics for risk-sensitive route planning (Experiment 4).

Measures how well different scoring methods select routes that survive perturbation.
"""

import numpy as np
from typing import Optional


def route_success_rate(success_flags: list[bool]) -> float:
    """Fraction of routes that successfully produce product."""
    if not success_flags:
        return 0.0
    return sum(success_flags) / len(success_flags)


def mean_fallback_depth(step_outcomes: list[list[bool]]) -> float:
    """
    Average number of steps attempted before first failure.
    Lower is worse (fails early). Max = route length (all succeed).
    """
    depths = []
    for route_steps in step_outcomes:
        depth = 0
        for success in route_steps:
            if success:
                depth += 1
            else:
                break
        depths.append(depth)
    return np.mean(depths) if depths else 0.0


def cost_adjusted_success(
    success_rates: np.ndarray,
    costs: np.ndarray,
    cost_weight: float = 0.1,
) -> np.ndarray:
    """
    Score = success_rate - cost_weight * cost.
    Balances reliability against expense.
    """
    return success_rates - cost_weight * costs


def route_family_diversity(
    route_fingerprints: np.ndarray,
) -> float:
    """
    Measure diversity of a route portfolio using pairwise Tanimoto distance.
    Higher = more diverse fallback options.
    """
    n = len(route_fingerprints)
    if n < 2:
        return 0.0

    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = route_fingerprints[i], route_fingerprints[j]
            intersection = np.minimum(a, b).sum()
            union = np.maximum(a, b).sum()
            tanimoto = intersection / max(union, 1e-6)
            distances.append(1.0 - tanimoto)

    return float(np.mean(distances))


def scoring_method_comparison(
    methods_results: dict[str, dict],
) -> dict[str, dict[str, float]]:
    """
    Compare scoring methods on route selection quality.

    For each method, summarize:
    - top-5 average success rate under perturbation
    - top-5 average yield
    - top-5 worst-case yield (5th percentile)

    This is the key table for the paper.
    """
    summary = {}
    for method_name, results in methods_results.items():
        success_rates = results.get("top5_success_rates", [])
        mean_yields = results.get("top5_mean_yields", [])
        worst_yields = results.get("top5_worst_yields", [])

        summary[method_name] = {
            "avg_success_rate": float(np.mean(success_rates)) if success_rates else 0.0,
            "avg_mean_yield": float(np.mean(mean_yields)) if mean_yields else 0.0,
            "avg_worst_yield": float(np.mean(worst_yields)) if worst_yields else 0.0,
            "best_success_rate": float(np.max(success_rates)) if success_rates else 0.0,
        }

    return summary


def perturbation_survival_curve(
    perturbation_scales: list[float],
    success_rates_by_scale: dict[str, list[float]],
) -> dict[str, float]:
    """
    For each scoring method, compute the area under the perturbation survival curve.

    x-axis: perturbation magnitude
    y-axis: route success rate

    Larger AUC = more robust route selection.
    """
    results = {}
    for method, rates in success_rates_by_scale.items():
        auc = np.trapz(rates, perturbation_scales)
        results[f"{method}_survival_auc"] = float(auc)
    return results
