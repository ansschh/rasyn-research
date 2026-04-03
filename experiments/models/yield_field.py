"""
Yield-Feasibility Field Model — Predict yield and feasibility as a FIELD over condition space.

Instead of y = f(reaction), this models y = f(reaction, conditions) with:
- Heteroscedastic regression for yield (predict mean + variance)
- Calibrated sigmoid for feasibility/success probability
- Deep ensemble for epistemic uncertainty
- Gradient-based robustness analysis: how sensitive is yield to condition perturbation?

This is the core of Experiment 3.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class YieldFeasibilityField(nn.Module):
    """
    Single model mapping (reaction, condition) -> (yield, feasibility, uncertainty).

    Outputs:
    - yield_mean: expected yield in [0, 1]
    - yield_log_var: log-variance for heteroscedastic uncertainty
    - feasibility: P(success) as calibrated probability
    """

    def __init__(
        self,
        reaction_dim: int = 256,
        condition_dim: int = 128,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1,
        gradient_penalty_weight: float = 0.0,
    ):
        super().__init__()
        self.gradient_penalty_weight = gradient_penalty_weight

        # Shared trunk
        layers = []
        in_dim = reaction_dim + condition_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.trunk = nn.Sequential(*layers)

        # Yield head — heteroscedastic (predict mean and log-variance)
        self.yield_mean_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # yield in [0, 1]
        )
        self.yield_logvar_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Feasibility head — calibrated binary prediction
        self.feasibility_head = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        reaction_repr: torch.Tensor,    # (batch, reaction_dim)
        condition_repr: torch.Tensor,    # (batch, condition_dim)
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass: predict yield field and feasibility at given condition point.
        """
        joint = torch.cat([reaction_repr, condition_repr], dim=-1)
        h = self.trunk(joint)

        yield_mean = self.yield_mean_head(h).squeeze(-1)       # (batch,)
        yield_logvar = self.yield_logvar_head(h).squeeze(-1)   # (batch,)
        feasibility_logit = self.feasibility_head(h).squeeze(-1)  # (batch,)
        feasibility_prob = torch.sigmoid(feasibility_logit)

        return {
            "yield_mean": yield_mean,
            "yield_logvar": yield_logvar,
            "yield_std": torch.exp(0.5 * yield_logvar),
            "feasibility_logit": feasibility_logit,
            "feasibility_prob": feasibility_prob,
        }

    def compute_loss(
        self,
        predictions: dict,
        yield_target: torch.Tensor,      # (batch,) in [0, 1]
        has_yield: torch.Tensor,         # (batch,) mask
        success_target: torch.Tensor,    # (batch,) binary
        reaction_repr: Optional[torch.Tensor] = None,
        condition_repr: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute heteroscedastic Gaussian NLL for yield + BCE for feasibility.
        """
        losses = {}

        # Yield: Gaussian NLL with learned variance
        if has_yield.sum() > 0:
            mean = predictions["yield_mean"]
            logvar = predictions["yield_logvar"].clamp(-10, 10)  # prevent NaN from extreme logvar
            # NLL = 0.5 * (log(var) + (y - mean)^2 / var)
            nll = 0.5 * (logvar + (yield_target - mean) ** 2 * torch.exp(-logvar))
            yield_loss = (nll * has_yield).sum() / has_yield.sum().clamp(min=1)
            losses["yield_nll"] = yield_loss
        else:
            losses["yield_nll"] = torch.tensor(0.0, device=yield_target.device)

        # Feasibility: Binary cross entropy
        feas_loss = F.binary_cross_entropy_with_logits(
            predictions["feasibility_logit"], success_target, reduction="mean"
        )
        losses["feasibility_bce"] = feas_loss

        # Optional: gradient penalty for smooth fields
        if self.gradient_penalty_weight > 0 and condition_repr is not None and condition_repr.requires_grad:
            grad = torch.autograd.grad(
                predictions["yield_mean"].sum(), condition_repr,
                create_graph=True, retain_graph=True,
            )[0]
            gp = (grad.norm(dim=-1) ** 2).mean()
            losses["gradient_penalty"] = self.gradient_penalty_weight * gp
        else:
            losses["gradient_penalty"] = torch.tensor(0.0, device=yield_target.device)

        losses["total"] = losses["yield_nll"] + losses["feasibility_bce"] + losses["gradient_penalty"]
        return losses

    @torch.no_grad()
    def compute_robustness(
        self,
        reaction_repr: torch.Tensor,
        condition_repr: torch.Tensor,
        perturbation_scale: float = 0.1,
        n_perturbations: int = 50,
    ) -> dict[str, torch.Tensor]:
        """
        Estimate robustness by evaluating yield/feasibility at perturbed conditions.

        Returns:
        - yield_std_perturb: std of yield under perturbation
        - feasibility_drop: how much feasibility drops under worst perturbation
        - robustness_score: combined robustness metric
        """
        batch_size = reaction_repr.size(0)
        cond_dim = condition_repr.size(1)

        # Generate perturbations
        noise = torch.randn(batch_size, n_perturbations, cond_dim, device=condition_repr.device) * perturbation_scale
        perturbed = condition_repr.unsqueeze(1) + noise  # (batch, n_perturb, cond_dim)

        # Evaluate at all perturbed points
        rxn_exp = reaction_repr.unsqueeze(1).expand(-1, n_perturbations, -1)
        preds = self.forward(
            rxn_exp.reshape(-1, reaction_repr.size(-1)),
            perturbed.reshape(-1, cond_dim),
        )

        yield_vals = preds["yield_mean"].reshape(batch_size, n_perturbations)
        feas_vals = preds["feasibility_prob"].reshape(batch_size, n_perturbations)

        # Nominal predictions
        nominal = self.forward(reaction_repr, condition_repr)

        return {
            "yield_mean_perturbed": yield_vals.mean(dim=1),
            "yield_std_perturbed": yield_vals.std(dim=1),
            "feasibility_min_perturbed": feas_vals.min(dim=1).values,
            "feasibility_mean_perturbed": feas_vals.mean(dim=1),
            "nominal_yield": nominal["yield_mean"],
            "nominal_feasibility": nominal["feasibility_prob"],
            # Robustness = how much does the WORST perturbation hurt?
            "yield_worst_drop": nominal["yield_mean"] - yield_vals.min(dim=1).values,
            "feasibility_worst_drop": nominal["feasibility_prob"] - feas_vals.min(dim=1).values,
        }


class YieldFieldEnsemble(nn.Module):
    """
    Deep Ensemble of YieldFeasibilityField models for epistemic uncertainty.

    Each member is independently initialized and trained.
    Disagreement between members = epistemic uncertainty.
    Individual variance = aleatoric uncertainty.
    """

    def __init__(self, n_members: int = 5, **model_kwargs):
        super().__init__()
        self.n_members = n_members
        self.members = nn.ModuleList([
            YieldFeasibilityField(**model_kwargs) for _ in range(n_members)
        ])

    def forward(
        self,
        reaction_repr: torch.Tensor,
        condition_repr: torch.Tensor,
        member_idx: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through all ensemble members (or a specific one for training).

        During training: use member_idx to train one member at a time.
        During inference: aggregate all members.
        """
        if member_idx is not None:
            return self.members[member_idx](reaction_repr, condition_repr)

        # Inference: run all members and aggregate
        all_preds = [m(reaction_repr, condition_repr) for m in self.members]

        yield_means = torch.stack([p["yield_mean"] for p in all_preds])      # (n_members, batch)
        yield_logvars = torch.stack([p["yield_logvar"] for p in all_preds])
        feas_probs = torch.stack([p["feasibility_prob"] for p in all_preds])

        # Ensemble mean and uncertainty decomposition
        ensemble_yield_mean = yield_means.mean(dim=0)
        aleatoric_var = torch.exp(yield_logvars).mean(dim=0)         # mean of individual variances
        epistemic_var = yield_means.var(dim=0)                        # variance of means
        total_var = aleatoric_var + epistemic_var

        ensemble_feas = feas_probs.mean(dim=0)
        feas_epistemic = feas_probs.var(dim=0)

        return {
            "yield_mean": ensemble_yield_mean,
            "yield_std": total_var.sqrt(),
            "yield_aleatoric_std": aleatoric_var.sqrt(),
            "yield_epistemic_std": epistemic_var.sqrt(),
            "feasibility_prob": ensemble_feas,
            "feasibility_epistemic_std": feas_epistemic.sqrt(),
        }

    def compute_robustness(
        self,
        reaction_repr: torch.Tensor,
        condition_repr: torch.Tensor,
        perturbation_scale: float = 0.1,
        n_perturbations: int = 50,
    ) -> dict[str, torch.Tensor]:
        """Compute robustness using ensemble predictions at perturbed conditions."""
        batch_size = reaction_repr.size(0)
        cond_dim = condition_repr.size(1)

        noise = torch.randn(batch_size, n_perturbations, cond_dim, device=condition_repr.device) * perturbation_scale
        perturbed = condition_repr.unsqueeze(1) + noise

        rxn_exp = reaction_repr.unsqueeze(1).expand(-1, n_perturbations, -1)
        preds = self.forward(
            rxn_exp.reshape(-1, reaction_repr.size(-1)),
            perturbed.reshape(-1, cond_dim),
        )

        yield_vals = preds["yield_mean"].reshape(batch_size, n_perturbations)
        feas_vals = preds["feasibility_prob"].reshape(batch_size, n_perturbations)
        uncert_vals = preds["yield_epistemic_std"].reshape(batch_size, n_perturbations)

        nominal = self.forward(reaction_repr, condition_repr)

        # Plateau width: fraction of perturbations where yield stays within 10% of nominal
        yield_threshold = nominal["yield_mean"].unsqueeze(1) * 0.9
        plateau_width = (yield_vals >= yield_threshold).float().mean(dim=1)

        return {
            "nominal_yield": nominal["yield_mean"],
            "nominal_feasibility": nominal["feasibility_prob"],
            "nominal_uncertainty": nominal["yield_epistemic_std"],
            "yield_mean_perturbed": yield_vals.mean(dim=1),
            "yield_std_perturbed": yield_vals.std(dim=1),
            "feasibility_min_perturbed": feas_vals.min(dim=1).values,
            "uncertainty_mean_perturbed": uncert_vals.mean(dim=1),
            "plateau_width": plateau_width,
            # Novel metric: robustness-adjusted utility
            "robust_utility": nominal["yield_mean"] * plateau_width,
        }
