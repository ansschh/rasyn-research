"""
Next-Measurement Policy — Active spectroscopic reasoning.

Given a current posterior over structure candidates, recommends which
measurement to acquire next to maximally reduce structural ambiguity.

This is the most novel component (H5) — almost absent in current literature.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


MEASUREMENT_ACTIONS = {
    0: "1H_NMR",
    1: "13C_NMR",
    2: "HSQC",
    3: "COSY",
    4: "MS_low_energy",
    5: "MS_high_energy",
}


class MeasurementPolicy(nn.Module):
    """
    Learned or heuristic policy for selecting the next measurement.

    Methods:
    - random: uniform random selection
    - fixed_order: always 1H -> 13C -> MS -> HSQC
    - entropy_greedy: pick measurement that maximizes expected entropy reduction
    - info_gain: full expected information gain (Bayesian experimental design)
    """

    def __init__(
        self,
        method: str = "entropy_greedy",
        n_actions: int = 6,
        posterior_dim: int = 256,
    ):
        super().__init__()
        self.method = method
        self.n_actions = n_actions

        # For learned policy
        if method in ("entropy_greedy", "info_gain"):
            self.action_scorer = nn.Sequential(
                nn.Linear(posterior_dim + n_actions, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
            )

    def select_action(
        self,
        posterior: torch.Tensor,             # (n_candidates,) current posterior
        available_actions: list[int],        # indices of measurements not yet taken
        candidate_mol_reprs: Optional[torch.Tensor] = None,  # (n_candidates, mol_dim)
        forward_models: Optional[dict] = None,  # forward models for simulation
    ) -> int:
        """Select the next measurement to acquire."""
        if self.method == "random":
            return self._random(available_actions)
        elif self.method == "fixed_order":
            return self._fixed_order(available_actions)
        elif self.method == "entropy_greedy":
            return self._entropy_greedy(posterior, available_actions, candidate_mol_reprs, forward_models)
        elif self.method == "info_gain":
            return self._info_gain(posterior, available_actions, candidate_mol_reprs, forward_models)
        else:
            return self._random(available_actions)

    def _random(self, available_actions: list[int]) -> int:
        return np.random.choice(available_actions)

    def _fixed_order(self, available_actions: list[int]) -> int:
        priority = [0, 1, 4, 2, 3, 5]  # 1H -> 13C -> MS_low -> HSQC -> COSY -> MS_high
        for a in priority:
            if a in available_actions:
                return a
        return available_actions[0]

    def _entropy_greedy(
        self,
        posterior: torch.Tensor,
        available_actions: list[int],
        candidate_mol_reprs: Optional[torch.Tensor],
        forward_models: Optional[dict],
    ) -> int:
        """
        Greedy entropy reduction.

        For each possible measurement, simulate what the posterior would look like
        after observing the measurement, then pick the one that reduces entropy most.
        """
        current_entropy = -(posterior * torch.log(posterior + 1e-10)).sum().item()

        best_action = available_actions[0]
        best_reduction = -float("inf")

        for action in available_actions:
            # Estimate expected posterior entropy after this measurement
            # (simplified: use forward model prediction diversity as proxy)
            expected_entropy = self._estimate_post_measurement_entropy(
                posterior, action, candidate_mol_reprs, forward_models
            )
            reduction = current_entropy - expected_entropy

            if reduction > best_reduction:
                best_reduction = reduction
                best_action = action

        return best_action

    def _info_gain(
        self,
        posterior: torch.Tensor,
        available_actions: list[int],
        candidate_mol_reprs: Optional[torch.Tensor],
        forward_models: Optional[dict],
    ) -> int:
        """
        Full expected information gain (Bayesian experimental design).

        IG(action) = H[posterior] - E_{x' ~ p(x'|action)} [H[posterior | x']]

        This requires integrating over possible measurement outcomes.
        """
        # For now, falls back to entropy greedy
        # Full implementation would Monte Carlo sample measurement outcomes
        return self._entropy_greedy(posterior, available_actions, candidate_mol_reprs, forward_models)

    def _estimate_post_measurement_entropy(
        self,
        posterior: torch.Tensor,
        action: int,
        candidate_mol_reprs: Optional[torch.Tensor],
        forward_models: Optional[dict],
    ) -> float:
        """
        Estimate posterior entropy after acquiring a measurement.

        Simplified approach:
        - Use forward model to predict what each candidate would produce
        - Candidates with similar predictions remain ambiguous
        - Candidates with different predictions get separated

        The action that produces the most diverse predictions across candidates
        will reduce entropy the most.
        """
        if candidate_mol_reprs is None:
            return 0.0

        n_cand = candidate_mol_reprs.size(0)

        # Use representation diversity as proxy for information value
        # Different measurements reveal different aspects of molecular structure
        # (This is a simplified heuristic — the full version would use forward models)

        # Hash the action into a projection direction
        torch.manual_seed(action * 12345)
        proj = torch.randn(candidate_mol_reprs.size(-1), device=candidate_mol_reprs.device)

        # Project candidates
        projected = candidate_mol_reprs @ proj
        weighted_mean = (posterior * projected).sum()
        weighted_var = (posterior * (projected - weighted_mean) ** 2).sum()

        # Higher variance → measurement is more informative → lower post-entropy
        # (very rough heuristic)
        estimated_entropy = -(posterior * torch.log(posterior + 1e-10)).sum().item()
        reduction_factor = 1.0 / (1.0 + weighted_var.item())

        return estimated_entropy * reduction_factor


# ─── Simulation loop for measurement policy evaluation ────────────────────────

def simulate_measurement_sequence(
    policy: MeasurementPolicy,
    posterior_engine,
    candidate_graphs: list[dict],
    true_index: int,
    all_spectra: dict[int, dict],  # action_id -> {"peaks": tensor, "mask": tensor}
    max_measurements: int = 4,
) -> dict:
    """
    Simulate a sequential measurement workflow.

    Start with no observations, iteratively:
    1. Select next measurement via policy
    2. "Acquire" it (reveal the spectrum)
    3. Update posterior
    4. Check if we've identified the structure

    Returns:
        history: measurements taken, posteriors, entropy at each step
    """
    n_actions = len(MEASUREMENT_ACTIONS)
    available = list(range(n_actions))
    acquired = {}
    history = {
        "actions": [],
        "entropy": [],
        "rank": [],
        "top1": [],
    }

    posterior = torch.ones(len(candidate_graphs)) / len(candidate_graphs)

    for step in range(max_measurements):
        if not available:
            break

        # Select next measurement
        action = policy.select_action(posterior, available)
        available.remove(action)
        history["actions"].append(action)

        # "Acquire" the measurement
        if action in all_spectra:
            acquired[action] = all_spectra[action]

        # Update posterior (re-run inference with all acquired spectra)
        # Simplified: accumulate evidence
        nmr_peaks, nmr_mask = None, None
        msms_peaks, msms_mask = None, None

        for act_id, spec in acquired.items():
            if act_id in [0, 1, 2, 3]:  # NMR-type
                nmr_peaks = spec["peaks"]
                nmr_mask = spec["mask"]
            else:  # MS-type
                msms_peaks = spec["peaks"]
                msms_mask = spec["mask"]

        with torch.no_grad():
            result = posterior_engine.compute_posterior(
                candidate_graphs,
                nmr_peaks=nmr_peaks, nmr_mask=nmr_mask,
                msms_peaks=msms_peaks, msms_mask=msms_mask,
            )
            posterior = result["posterior"]

        # Record metrics
        entropy = result["entropy"].item()
        rank = (posterior > posterior[true_index]).sum().item() + 1

        history["entropy"].append(entropy)
        history["rank"].append(rank)
        history["top1"].append(float(rank == 1))

    return history


def compare_policies(
    policies: dict[str, MeasurementPolicy],
    posterior_engine,
    test_molecules: list[dict],
    n_candidates: int = 50,
    max_measurements: int = 4,
) -> dict[str, dict]:
    """
    Compare measurement policies on a set of test molecules.

    Returns average metrics per policy.
    """
    results = {}
    for policy_name, policy in policies.items():
        all_histories = []

        for mol_data in test_molecules:
            history = simulate_measurement_sequence(
                policy, posterior_engine,
                mol_data["candidates"],
                mol_data["true_index"],
                mol_data["all_spectra"],
                max_measurements=max_measurements,
            )
            all_histories.append(history)

        # Aggregate
        results[policy_name] = {
            "avg_final_rank": np.mean([h["rank"][-1] for h in all_histories]),
            "avg_final_top1": np.mean([h["top1"][-1] for h in all_histories]),
            "avg_measurements_to_top1": np.mean([
                next((i + 1 for i, t in enumerate(h["top1"]) if t == 1.0), max_measurements)
                for h in all_histories
            ]),
            "avg_entropy_reduction": np.mean([
                h["entropy"][0] - h["entropy"][-1] if h["entropy"] else 0
                for h in all_histories
            ]),
        }

    return results
