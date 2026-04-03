"""
Experiment 4: Route Scoring Comparison.

Compare different route scoring methods on synthetic routes:
1. Expected yield (baseline)
2. Worst step
3. CVaR (5% and 10%)
4. Chance-constrained
5. Robustness-window

For each method, rank routes by score, then simulate perturbation
to see which method's top routes actually survive.
"""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.route_scorer import (
    Route, RouteStep, RouteScorer,
    simulate_route_under_perturbation,
    compare_scoring_methods,
)


def generate_synthetic_routes(n_targets: int = 200, n_routes_per_target: int = 20, seed: int = 42) -> list[Route]:
    """
    Generate synthetic routes with varying yield/robustness profiles.

    Creates routes that test the key hypothesis: robustness-aware scoring
    should prefer routes with broad condition windows over routes with
    slightly higher but fragile yields.
    """
    rng = np.random.RandomState(seed)
    routes = []

    for t in range(n_targets):
        for r in range(n_routes_per_target):
            n_steps = rng.randint(2, 6)
            steps = []

            # Route archetype: mix of robust and fragile steps
            route_type = rng.choice(["robust", "fragile", "mixed"], p=[0.3, 0.3, 0.4])

            for s in range(n_steps):
                if route_type == "robust":
                    # Lower yield but wide plateau
                    yield_mean = rng.uniform(0.5, 0.75)
                    yield_std = rng.uniform(0.02, 0.08)
                    feasibility_prob = rng.uniform(0.85, 0.98)
                    feasibility_std = rng.uniform(0.01, 0.05)
                    plateau_width = rng.uniform(0.7, 0.95)
                    condition_window = rng.uniform(0.5, 0.9)
                elif route_type == "fragile":
                    # Higher yield but narrow peak
                    yield_mean = rng.uniform(0.7, 0.95)
                    yield_std = rng.uniform(0.10, 0.25)
                    feasibility_prob = rng.uniform(0.6, 0.85)
                    feasibility_std = rng.uniform(0.05, 0.15)
                    plateau_width = rng.uniform(0.1, 0.4)
                    condition_window = rng.uniform(0.1, 0.3)
                else:
                    # Mix
                    yield_mean = rng.uniform(0.4, 0.9)
                    yield_std = rng.uniform(0.03, 0.15)
                    feasibility_prob = rng.uniform(0.65, 0.95)
                    feasibility_std = rng.uniform(0.02, 0.10)
                    plateau_width = rng.uniform(0.3, 0.8)
                    condition_window = rng.uniform(0.2, 0.7)

                steps.append(RouteStep(
                    step_id=s,
                    product_smiles=f"target_{t}_step_{s}_product",
                    reactant_smiles=[f"target_{t}_step_{s}_reactant"],
                    yield_mean=yield_mean,
                    yield_std=yield_std,
                    feasibility_prob=feasibility_prob,
                    feasibility_std=feasibility_std,
                    plateau_width=plateau_width,
                    yield_worst_drop=yield_std * 2,
                    condition_window_volume=condition_window,
                ))

            routes.append(Route(
                route_id=f"target_{t}_route_{r}",
                target_smiles=f"target_{t}",
                steps=steps,
            ))

    return routes


def main():
    print("=" * 60)
    print("  Experiment 4: Route Scoring Comparison")
    print("=" * 60)

    # Generate routes
    print("\nGenerating synthetic routes...")
    routes = generate_synthetic_routes(n_targets=200, n_routes_per_target=20)
    print(f"  Generated {len(routes)} routes for {200} targets")

    # Compare scoring methods
    print("\nComparing scoring methods under perturbation...")
    results = compare_scoring_methods(routes, perturbation_kwargs={"n_trials": 200})

    # Print results table
    print("\n" + "=" * 80)
    print(f"{'Method':<25} {'Avg Success Rate':>18} {'Avg Mean Yield':>16} {'Avg Worst Yield':>17}")
    print("-" * 80)

    for method, metrics in results.items():
        sr = np.mean(metrics["top5_success_rates"])
        my = np.mean(metrics["top5_mean_yields"])
        wy = np.mean(metrics["top5_worst_yields"])
        print(f"{method:<25} {sr:>18.4f} {my:>16.4f} {wy:>17.4f}")

    print("=" * 80)

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert to serializable
    serializable = {}
    for method, metrics in results.items():
        serializable[method] = {k: [float(v) for v in vals] if isinstance(vals, list) else vals
                                for k, vals in metrics.items()}

    with open(results_dir / "route_scoring_comparison.json", "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {results_dir / 'route_scoring_comparison.json'}")

    # Key finding
    print("\n--- Key Finding ---")
    methods_by_success = sorted(results.items(),
                                 key=lambda x: np.mean(x[1]["top5_success_rates"]),
                                 reverse=True)
    best_method = methods_by_success[0][0]
    best_sr = np.mean(methods_by_success[0][1]["top5_success_rates"])
    baseline_sr = np.mean(results.get("expected_yield", {}).get("top5_success_rates", [0]))
    print(f"Best method: {best_method} (success rate: {best_sr:.4f})")
    print(f"Baseline (expected_yield): {baseline_sr:.4f}")
    print(f"Improvement: {(best_sr - baseline_sr) / max(baseline_sr, 0.001) * 100:.1f}%")


if __name__ == "__main__":
    main()
