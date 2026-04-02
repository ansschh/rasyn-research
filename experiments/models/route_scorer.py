"""
Risk-Sensitive Route Scorer — Score retrosynthetic routes using robust objectives.

Instead of: route_score = product(expected_yields)
We use: CVaR, chance-constrained, and robustness-window-aware scoring.

This is the core of Experiment 4.
"""

from typing import Optional
from dataclasses import dataclass, field

import torch
import numpy as np


@dataclass
class RouteStep:
    """A single step in a retrosynthetic route."""
    step_id: int
    product_smiles: str
    reactant_smiles: list[str]

    # Predicted outcomes (from yield field model)
    yield_mean: float = 0.0
    yield_std: float = 0.0
    feasibility_prob: float = 0.0
    feasibility_std: float = 0.0

    # Robustness metrics
    plateau_width: float = 0.0    # fraction of perturbations that maintain yield
    yield_worst_drop: float = 0.0  # worst yield drop under perturbation

    # Condition manifold metrics
    condition_window_volume: float = 0.0  # size of viable condition region


@dataclass
class Route:
    """A full retrosynthetic route."""
    route_id: str
    target_smiles: str
    steps: list[RouteStep] = field(default_factory=list)


class RouteScorer:
    """
    Score routes using different objectives.

    Methods:
    1. expected_yield: product of expected yields (baseline)
    2. worst_step: minimum yield/feasibility across steps
    3. cvar: Conditional Value at Risk — average over worst alpha-fraction of outcomes
    4. chance_constrained: maximize yield subject to P(all steps succeed) >= threshold
    5. robustness_window: score by average condition window volume
    """

    def __init__(self, method: str = "expected_yield", **kwargs):
        self.method = method
        self.kwargs = kwargs

    def score(self, route: Route) -> float:
        """Score a route using the configured method."""
        scorer = getattr(self, f"_score_{self.method}")
        return scorer(route)

    def _score_expected_yield(self, route: Route) -> float:
        """Baseline: product of expected yields across steps."""
        if not route.steps:
            return 0.0
        overall = 1.0
        for step in route.steps:
            overall *= step.yield_mean
        return overall

    def _score_worst_step(self, route: Route) -> float:
        """Score by the weakest link (minimum yield or feasibility)."""
        if not route.steps:
            return 0.0
        return min(step.yield_mean * step.feasibility_prob for step in route.steps)

    def _score_cvar(self, route: Route) -> float:
        """
        CVaR (Conditional Value at Risk) route scoring.

        Simulate route outcomes via Monte Carlo, then compute the average
        over the worst alpha-fraction of outcomes.
        """
        alpha = self.kwargs.get("alpha", 0.1)
        n_samples = self.kwargs.get("n_samples", 1000)

        if not route.steps:
            return 0.0

        # Monte Carlo simulation of route outcomes
        route_yields = np.ones(n_samples)
        for step in route.steps:
            # Sample yield per step from N(mean, std)
            step_yields = np.random.normal(step.yield_mean, step.yield_std, n_samples)
            step_yields = np.clip(step_yields, 0, 1)

            # Sample feasibility (Bernoulli)
            step_success = np.random.random(n_samples) < step.feasibility_prob
            step_outcomes = step_yields * step_success

            route_yields *= step_outcomes

        # CVaR: average of bottom alpha-fraction
        sorted_yields = np.sort(route_yields)
        cutoff = int(n_samples * alpha)
        if cutoff == 0:
            cutoff = 1
        cvar = sorted_yields[:cutoff].mean()

        return float(cvar)

    def _score_chance_constrained(self, route: Route) -> float:
        """
        Chance-constrained scoring.

        Score = expected_yield if P(all steps succeed) >= threshold, else penalized.
        """
        threshold = self.kwargs.get("success_threshold", 0.8)

        if not route.steps:
            return 0.0

        # P(route succeeds) ≈ product of step feasibilities (independence assumption)
        route_success_prob = 1.0
        for step in route.steps:
            route_success_prob *= step.feasibility_prob

        expected = self._score_expected_yield(route)

        if route_success_prob >= threshold:
            return expected
        else:
            # Penalize proportionally to constraint violation
            penalty = (threshold - route_success_prob) / threshold
            return expected * (1.0 - penalty)

    def _score_robustness_window(self, route: Route) -> float:
        """
        Score by average robustness of the condition windows.

        Routes where every step has a wide viable condition region are preferred.
        """
        if not route.steps:
            return 0.0

        expected = self._score_expected_yield(route)
        avg_plateau = np.mean([step.plateau_width for step in route.steps])
        avg_window = np.mean([step.condition_window_volume for step in route.steps])

        # Combined: yield * robustness
        return expected * avg_plateau * (1.0 + avg_window)


# ─── Route perturbation simulation ───────────────────────────────────────────

def simulate_route_under_perturbation(
    route: Route,
    n_trials: int = 100,
    temp_noise_std: float = 10.0,
    yield_noise_scale: float = 0.1,
) -> dict[str, float]:
    """
    Simulate route execution under realistic perturbations.

    For each trial:
    - Perturb conditions slightly
    - Resample yields and feasibility
    - Check if route still produces product

    Returns:
        success_rate: fraction of trials where route succeeds
        mean_yield: average yield across successful trials
        worst_yield: 5th percentile yield
    """
    successes = 0
    yields = []

    for _ in range(n_trials):
        route_yield = 1.0
        route_succeeds = True

        for step in route.steps:
            # Perturbed yield = nominal + noise proportional to (1 - plateau_width)
            vulnerability = 1.0 - step.plateau_width
            perturbed_yield = step.yield_mean + np.random.normal(0, step.yield_std + yield_noise_scale * vulnerability)
            perturbed_yield = np.clip(perturbed_yield, 0, 1)

            # Perturbed feasibility
            perturbed_feas = step.feasibility_prob - np.abs(np.random.normal(0, step.feasibility_std * 0.5))
            step_succeeds = np.random.random() < max(perturbed_feas, 0)

            if not step_succeeds:
                route_succeeds = False
                break

            route_yield *= perturbed_yield

        if route_succeeds:
            successes += 1
            yields.append(route_yield)

    success_rate = successes / n_trials
    mean_yield = np.mean(yields) if yields else 0.0
    worst_yield = np.percentile(yields, 5) if len(yields) >= 20 else 0.0

    return {
        "success_rate": success_rate,
        "mean_yield": mean_yield,
        "worst_yield_5pct": worst_yield,
        "n_successful": successes,
    }


def compare_scoring_methods(
    routes: list[Route],
    perturbation_kwargs: Optional[dict] = None,
) -> dict[str, list]:
    """
    Compare different route scoring methods on the same set of routes.

    For each method, rank routes by score, then measure how well the
    top-ranked routes actually perform under perturbation.
    """
    if perturbation_kwargs is None:
        perturbation_kwargs = {}

    methods = [
        ("expected_yield", {}),
        ("worst_step", {}),
        ("cvar_05", {"alpha": 0.05}),
        ("cvar_10", {"alpha": 0.10}),
        ("chance_constrained", {"success_threshold": 0.8}),
        ("robustness_window", {}),
    ]

    results = {}
    for method_name, method_kwargs in methods:
        scorer = RouteScorer(
            method=method_name.split("_")[0] if method_name.startswith("cvar") else method_name,
            **method_kwargs,
        )

        # Score and rank
        scores = [(route, scorer.score(route)) for route in routes]
        scores.sort(key=lambda x: x[1], reverse=True)

        # Evaluate top-5 under perturbation
        top_routes = [r for r, s in scores[:5]]
        perturb_results = [simulate_route_under_perturbation(r, **perturbation_kwargs) for r in top_routes]

        results[method_name] = {
            "top5_scores": [s for _, s in scores[:5]],
            "top5_success_rates": [r["success_rate"] for r in perturb_results],
            "top5_mean_yields": [r["mean_yield"] for r in perturb_results],
            "top5_worst_yields": [r["worst_yield_5pct"] for r in perturb_results],
        }

    return results
