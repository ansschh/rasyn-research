"""
Experiment 5: Boundary-Focused Active Learning.

Compare acquisition strategies on Doyle HTE data:
1. Random
2. Uncertainty sampling
3. Boundary gradient (our method)
4. Boundary uncertainty (our method)

Uses Doyle HTE as oracle (dense condition grid with known yields).
"""

import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.yield_field import YieldFeasibilityField
from models.active_learner import BoundaryAcquisition, run_active_learning_loop


def create_synthetic_hte_oracle(n_conditions: int = 500, n_dims: int = 8, seed: int = 42):
    """
    Create a synthetic HTE oracle with known viability boundaries.

    Simulates a reaction where yield depends on conditions in a nonlinear way,
    with a clear viable/non-viable boundary.
    """
    rng = np.random.RandomState(seed)

    # Generate condition space
    conditions = rng.randn(n_conditions, n_dims).astype(np.float32)

    # True yield function: combination of Gaussians creating a viability region
    center1 = rng.randn(n_dims).astype(np.float32) * 0.5
    center2 = rng.randn(n_dims).astype(np.float32) * 0.5

    dist1 = np.sum((conditions - center1) ** 2, axis=1)
    dist2 = np.sum((conditions - center2) ** 2, axis=1)

    # Yield = mixture of Gaussians (creates regions of high/low yield)
    yields = 0.7 * np.exp(-dist1 / 4.0) + 0.5 * np.exp(-dist2 / 3.0)
    yields = np.clip(yields + rng.randn(n_conditions) * 0.05, 0, 1).astype(np.float32)

    # Success = yield > threshold
    success = (yields > 0.3).astype(np.float32)

    return conditions, yields, success


def simple_train_fn(model, reaction_repr, X_train, y_train, s_train):
    """Simple training function for the active learning loop."""
    device = next(model.parameters()).device if hasattr(model, 'parameters') else torch.device('cpu')

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    rxn = reaction_repr.to(device)
    X = X_train.to(device)
    y = y_train.to(device)
    s = s_train.to(device)

    n = len(X)
    batch_size = min(64, n)

    for epoch in range(50):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            rxn_batch = rxn.expand(len(idx), -1)

            preds = model(rxn_batch, X[idx])

            # Simple MSE + BCE loss
            yield_loss = ((preds["yield_mean"] - y[idx]) ** 2).mean()
            feas_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                preds["feasibility_logit"], s[idx]
            )

            loss = yield_loss + feas_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def main():
    print("=" * 60)
    print("  Experiment 5: Boundary-Focused Active Learning")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create oracle
    print("\nCreating synthetic HTE oracle...")
    conditions, yields, success = create_synthetic_hte_oracle(n_conditions=500, n_dims=8)

    conditions_t = torch.tensor(conditions)
    yields_t = torch.tensor(yields)
    success_t = torch.tensor(success)

    # Fixed reaction representation (same reaction, varying conditions)
    reaction_repr = torch.randn(1, 256)

    print(f"  Conditions: {conditions.shape}")
    print(f"  Yield range: {yields.min():.3f} - {yields.max():.3f}")
    print(f"  Success rate: {success.mean():.3f}")

    # Model config
    model_kwargs = {
        "reaction_dim": 256,
        "condition_dim": 8,
        "hidden_dims": [128, 64],
        "dropout": 0.1,
    }

    # Run active learning for each acquisition method
    methods = ["random", "uncertainty", "boundary_gradient", "boundary_uncertainty"]
    all_results = {}

    for method in methods:
        print(f"\n--- Running: {method} ---")

        method_histories = []
        for repeat in range(5):
            history = run_active_learning_loop(
                model_class=YieldFeasibilityField,
                model_kwargs=model_kwargs,
                train_fn=simple_train_fn,
                reaction_repr=reaction_repr,
                all_conditions=conditions_t,
                all_yields=yields_t,
                all_success=success_t,
                initial_fraction=0.10,
                budget_per_round=20,
                n_rounds=15,
                acquisition_method=method,
                seed=42 + repeat,
            )
            method_histories.append(history)
            print(f"  Repeat {repeat+1}: final IoU={history['viability_iou'][-1]:.3f}, "
                  f"AUROC={history['feasibility_auroc'][-1]:.3f}")

        # Average across repeats
        avg_history = {}
        for key in method_histories[0]:
            if key == "round":
                avg_history[key] = method_histories[0][key]
            else:
                vals = np.array([h[key] for h in method_histories])
                avg_history[f"{key}_mean"] = vals.mean(axis=0).tolist()
                avg_history[f"{key}_std"] = vals.std(axis=0).tolist()

        all_results[method] = avg_history

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Method':<25} {'Final IoU':>12} {'Final AUROC':>13} {'Experiments to 0.8 IoU':>23}")
    print("-" * 80)

    for method, res in all_results.items():
        final_iou = res["viability_iou_mean"][-1]
        final_auroc = res["feasibility_auroc_mean"][-1]

        # Find experiments to reach 0.8 IoU
        iou_vals = res["viability_iou_mean"]
        n_labeled = res["n_labeled_mean"]
        reached = False
        for i, iou in enumerate(iou_vals):
            if iou >= 0.8:
                exp_to_target = n_labeled[i]
                reached = True
                break
        if not reached:
            exp_to_target = float("inf")

        print(f"{method:<25} {final_iou:>12.3f} {final_auroc:>13.3f} {exp_to_target:>23.0f}")

    print("=" * 80)

    # Save results
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "active_learning_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nResults saved to {results_dir / 'active_learning_comparison.json'}")


if __name__ == "__main__":
    main()
