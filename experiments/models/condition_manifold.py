"""
Condition Manifold Model — Energy-Based Model for set-valued condition prediction.

Instead of predicting a single condition tuple, this model learns an energy function
E(T, c) where low-energy regions define the SET of viable conditions for reaction T.

This is the core novel contribution of Experiment 2.

Key ideas:
- Energy-based model: E_phi(reaction, conditions) -> scalar energy
- Low energy = viable condition combination
- Level sets at threshold alpha define the predicted viable region
- Training via Noise Contrastive Estimation (NCE)
- Sampling via Stochastic Gradient Langevin Dynamics (SGLD)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConditionEmbedder(nn.Module):
    """Embed mixed discrete-continuous conditions into a single vector."""

    def __init__(
        self,
        discrete_vocab_sizes: dict[str, int],
        n_continuous: int = 3,
        embed_dim: int = 32,
        output_dim: int = 128,
    ):
        super().__init__()
        self.discrete_fields = list(discrete_vocab_sizes.keys())
        self.n_continuous = n_continuous

        # Learned embeddings for each discrete condition
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            for name, vocab_size in discrete_vocab_sizes.items()
        })

        # Project continuous conditions
        self.continuous_proj = nn.Linear(n_continuous, embed_dim)

        # Combine all condition embeddings
        total_input = embed_dim * (len(discrete_vocab_sizes) + 1)  # +1 for continuous
        self.combine = nn.Sequential(
            nn.Linear(total_input, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(
        self,
        discrete_ids: torch.LongTensor,    # (batch, n_discrete)
        continuous_vals: torch.FloatTensor, # (batch, n_continuous)
    ) -> torch.Tensor:
        """Embed conditions into a fixed-size vector."""
        parts = []
        for i, name in enumerate(self.discrete_fields):
            emb = self.embeddings[name](discrete_ids[:, i])  # (batch, embed_dim)
            parts.append(emb)

        cont_emb = self.continuous_proj(continuous_vals)  # (batch, embed_dim)
        parts.append(cont_emb)

        combined = torch.cat(parts, dim=-1)
        return self.combine(combined)


class ConditionManifoldEBM(nn.Module):
    """
    Energy-Based Model for set-valued condition prediction.

    Given a reaction representation and a candidate condition vector,
    outputs a scalar energy. Low energy = viable condition combination.

    The viable condition SET for a reaction is defined as:
        V_alpha(T) = { c : E(T, c) < threshold_alpha }

    Training: Noise Contrastive Estimation
    - Positive: observed (reaction, condition) pairs
    - Negative: corrupted conditions (swap discrete components, add noise to continuous)

    Inference: Sample from the low-energy region via SGLD
    """

    def __init__(
        self,
        reaction_dim: int = 256,
        condition_dim: int = 128,
        hidden_dims: list[int] = [512, 256, 128],
        discrete_vocab_sizes: dict[str, int] = None,
        n_continuous: int = 3,
        embed_dim: int = 32,
    ):
        super().__init__()

        if discrete_vocab_sizes is None:
            discrete_vocab_sizes = {
                "solvent": 200, "catalyst": 200,
                "reagent": 300, "base": 100, "ligand": 100,
            }

        self.condition_embedder = ConditionEmbedder(
            discrete_vocab_sizes, n_continuous, embed_dim, condition_dim
        )

        # Energy network: (reaction_repr, condition_repr) -> scalar energy
        layers = []
        in_dim = reaction_dim + condition_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        self.energy_net = nn.Sequential(*layers)

        # Pairwise compatibility energy between discrete conditions
        # This captures interactions (e.g., certain solvents work only with certain catalysts)
        self.compatibility = nn.Bilinear(condition_dim, reaction_dim, 1, bias=False)

    def energy(
        self,
        reaction_repr: torch.Tensor,        # (batch, reaction_dim)
        discrete_ids: torch.LongTensor,      # (batch, n_discrete)
        continuous_vals: torch.FloatTensor,   # (batch, n_continuous)
    ) -> torch.Tensor:
        """
        Compute energy E(T, c) for a (reaction, condition) pair.
        Lower energy = more viable condition.

        Returns: (batch,) energy values
        """
        cond_repr = self.condition_embedder(discrete_ids, continuous_vals)

        # Main energy from joint representation
        joint = torch.cat([reaction_repr, cond_repr], dim=-1)
        e_main = self.energy_net(joint).squeeze(-1)

        # Compatibility energy (bilinear interaction)
        e_compat = self.compatibility(cond_repr, reaction_repr).squeeze(-1)

        return e_main + e_compat

    def forward(
        self,
        reaction_repr: torch.Tensor,
        discrete_ids: torch.LongTensor,
        continuous_vals: torch.FloatTensor,
    ) -> torch.Tensor:
        """Alias for energy()."""
        return self.energy(reaction_repr, discrete_ids, continuous_vals)

    def nce_loss(
        self,
        reaction_repr: torch.Tensor,
        pos_discrete: torch.LongTensor,
        pos_continuous: torch.FloatTensor,
        neg_discrete: torch.LongTensor,
        neg_continuous: torch.FloatTensor,
    ) -> dict[str, torch.Tensor]:
        """
        Noise Contrastive Estimation loss.

        Positive pairs should have lower energy than negative pairs.

        Args:
            reaction_repr: (batch, reaction_dim)
            pos_discrete, pos_continuous: observed conditions
            neg_discrete, neg_continuous: corrupted conditions (batch * n_neg, ...)
        """
        batch_size = reaction_repr.size(0)
        n_neg = neg_discrete.size(0) // batch_size

        # Positive energy
        e_pos = self.energy(reaction_repr, pos_discrete, pos_continuous)  # (batch,)

        # Negative energy — expand reaction_repr to match negatives
        rxn_expanded = reaction_repr.unsqueeze(1).expand(-1, n_neg, -1).reshape(-1, reaction_repr.size(-1))
        e_neg = self.energy(rxn_expanded, neg_discrete, neg_continuous)  # (batch * n_neg,)
        e_neg = e_neg.view(batch_size, n_neg)

        # NCE: positive should have lower energy
        # logits: (batch, 1 + n_neg) where first column is positive
        logits = torch.cat([-e_pos.unsqueeze(1), -e_neg], dim=1)  # higher logit = lower energy = more likely
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)  # positive is class 0

        loss = F.cross_entropy(logits, labels)

        # Accuracy for monitoring
        pred = logits.argmax(dim=1)
        acc = (pred == 0).float().mean()

        return {
            "nce_loss": loss,
            "nce_accuracy": acc,
            "energy_pos_mean": e_pos.mean(),
            "energy_neg_mean": e_neg.mean(),
            "energy_gap": (e_neg.mean() - e_pos.mean()),
        }

    @torch.no_grad()
    def sample_sgld(
        self,
        reaction_repr: torch.Tensor,
        n_samples: int = 100,
        n_steps: int = 20,
        step_size: float = 0.01,
        noise_scale: float = 0.005,
        discrete_vocab_sizes: Optional[dict[str, int]] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Sample conditions from the energy model via SGLD (continuous part only).

        For discrete conditions, we use Gibbs-like sampling:
        fix continuous, enumerate discrete, pick low-energy combos.

        Returns sampled conditions with their energies.
        """
        batch_size = reaction_repr.size(0)
        n_continuous = 3  # default

        # Initialize continuous from noise
        c_cont = torch.randn(batch_size, n_samples, n_continuous, device=reaction_repr.device) * 0.5

        # For discrete: start from random valid indices
        if discrete_vocab_sizes is None:
            n_discrete = len(self.condition_embedder.discrete_fields)
            c_disc = torch.randint(1, 50, (batch_size, n_samples, n_discrete), device=reaction_repr.device)
        else:
            fields = list(discrete_vocab_sizes.keys())
            disc_parts = []
            for name in fields:
                vs = discrete_vocab_sizes[name]
                disc_parts.append(torch.randint(1, vs, (batch_size, n_samples, 1), device=reaction_repr.device))
            c_disc = torch.cat(disc_parts, dim=-1)

        # SGLD on continuous conditions
        rxn_exp = reaction_repr.unsqueeze(1).expand(-1, n_samples, -1)

        for step in range(n_steps):
            c_cont_flat = c_cont.reshape(-1, n_continuous).detach().requires_grad_(True)
            c_disc_flat = c_disc.reshape(-1, c_disc.size(-1))
            rxn_flat = rxn_exp.reshape(-1, reaction_repr.size(-1))

            energy = self.energy(rxn_flat, c_disc_flat, c_cont_flat)
            grad = torch.autograd.grad(energy.sum(), c_cont_flat)[0]

            c_cont_flat = c_cont_flat - step_size * grad + noise_scale * torch.randn_like(c_cont_flat)
            c_cont = c_cont_flat.reshape(batch_size, n_samples, n_continuous).detach()

        # Compute final energies
        c_cont_flat = c_cont.reshape(-1, n_continuous)
        c_disc_flat = c_disc.reshape(-1, c_disc.size(-1))
        rxn_flat = rxn_exp.reshape(-1, reaction_repr.size(-1))
        energies = self.energy(rxn_flat, c_disc_flat, c_cont_flat)
        energies = energies.reshape(batch_size, n_samples)

        return {
            "discrete_samples": c_disc,      # (batch, n_samples, n_discrete)
            "continuous_samples": c_cont,     # (batch, n_samples, n_continuous)
            "energies": energies,             # (batch, n_samples)
        }


def generate_negative_samples(
    discrete_ids: torch.LongTensor,     # (batch, n_discrete)
    continuous_vals: torch.FloatTensor,  # (batch, n_continuous)
    n_negatives: int = 64,
    noise_scale: float = 0.1,
    discrete_vocab_sizes: Optional[list[int]] = None,
) -> tuple[torch.LongTensor, torch.FloatTensor]:
    """
    Generate negative (corrupted) condition samples for NCE training.

    Strategies:
    1. Swap discrete components randomly
    2. Add Gaussian noise to continuous components
    3. Mix-and-match from different reactions in the batch
    """
    batch_size = discrete_ids.size(0)
    n_discrete = discrete_ids.size(1)
    n_continuous = continuous_vals.size(1)

    neg_disc_list = []
    neg_cont_list = []

    for _ in range(n_negatives):
        # Strategy mix: 40% swap discrete, 30% noise continuous, 30% cross-batch
        strategy = torch.rand(1).item()

        if strategy < 0.4:
            # Swap one random discrete component with a random value
            neg_d = discrete_ids.clone()
            col = torch.randint(0, n_discrete, (batch_size,))
            if discrete_vocab_sizes:
                max_val = discrete_vocab_sizes[col[0].item()]
            else:
                max_val = 200
            new_vals = torch.randint(1, max_val, (batch_size,), device=discrete_ids.device)
            neg_d[torch.arange(batch_size), col] = new_vals
            neg_cont = continuous_vals + torch.randn_like(continuous_vals) * noise_scale * 0.5

        elif strategy < 0.7:
            # Keep discrete, corrupt continuous more aggressively
            neg_d = discrete_ids.clone()
            neg_cont = continuous_vals + torch.randn_like(continuous_vals) * noise_scale * 2.0

        else:
            # Cross-batch: shuffle conditions from other reactions
            perm = torch.randperm(batch_size, device=discrete_ids.device)
            neg_d = discrete_ids[perm]
            neg_cont = continuous_vals[perm] + torch.randn_like(continuous_vals) * noise_scale

        neg_disc_list.append(neg_d)
        neg_cont_list.append(neg_cont)

    neg_discrete = torch.cat(neg_disc_list, dim=0)    # (batch * n_neg, n_discrete)
    neg_continuous = torch.cat(neg_cont_list, dim=0)   # (batch * n_neg, n_continuous)

    return neg_discrete, neg_continuous
