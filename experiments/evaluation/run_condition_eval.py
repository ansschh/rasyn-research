"""
Experiment 2 Evaluation: Point vs EBM Condition Prediction.

Head-to-head comparison on ORDerly test set:
- Top-k exact tuple match (point model strength)
- Coverage@alpha (EBM strength — does the predicted region contain valid conditions?)
- Energy gap quality (how well does EBM separate viable from non-viable?)
- NCE accuracy
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.featurization import SmilesTokenizer, ConditionEncoder, ReactionConditionDataset, create_dataloader
from data.preprocessing import load_records
from data.splits import random_split
from models.reaction_encoder import ReactionEncoder
from models.condition_point import PointConditionPredictor
from models.condition_manifold import ConditionManifoldEBM, generate_negative_samples
from training.utils import get_device, load_checkpoint


def main():
    print("=" * 60)
    print("  Experiment 2: Point vs EBM Condition Evaluation")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Load data
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    df = load_records(processed_dir / "orderly.parquet")

    tokenizer = SmilesTokenizer.load(processed_dir / "smiles_vocab.json")
    with open(processed_dir / "condition_vocab.json") as f:
        discrete_vocabs = json.load(f)
    with open(processed_dir / "continuous_stats.json") as f:
        continuous_stats = json.load(f)
    cond_encoder = ConditionEncoder(discrete_vocabs, continuous_stats)

    # Same split as training
    splits = random_split(df, test_frac=0.1, val_frac=0.1, seed=42)
    test_ds = ReactionConditionDataset(splits["test"], tokenizer, cond_encoder, max_seq_len=512)
    test_loader = create_dataloader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    enc_cfg = {"d_model": 256, "n_layers": 4, "n_heads": 4, "d_ff": 1024, "dropout": 0.1, "max_seq_len": 512}
    discrete_vocab_sizes = {k: len(v) for k, v in discrete_vocabs.items()}

    # ─── Load Point Model ────────────────────────────────────────────
    print("\nLoading point condition model...")
    encoder_point = ReactionEncoder(
        vocab_size=len(tokenizer.vocab), d_model=enc_cfg["d_model"],
        n_layers=enc_cfg["n_layers"], n_heads=enc_cfg["n_heads"],
        d_ff=enc_cfg["d_ff"], dropout=0, max_seq_len=enc_cfg["max_seq_len"],
    )
    point_head = PointConditionPredictor(
        reaction_dim=enc_cfg["d_model"],
        discrete_vocab_sizes=discrete_vocab_sizes,
        n_continuous=cond_encoder.n_continuous,
    )

    ckpt_dir = Path(__file__).parent.parent / "checkpoints"
    point_ckpt = ckpt_dir / "condition_point_random_seed42" / "best.pt"
    if point_ckpt.exists():
        ckpt = torch.load(point_ckpt, map_location=device, weights_only=False)
        # The checkpoint has the full ModelBundle state
        state = ckpt["model_state_dict"]
        enc_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
        head_state = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
        encoder_point.load_state_dict(enc_state)
        point_head.load_state_dict(head_state)
        print(f"  Loaded from {point_ckpt}")
    else:
        print(f"  WARNING: No checkpoint at {point_ckpt}")

    encoder_point.to(device).eval()
    point_head.to(device).eval()

    # ─── Load EBM Model ──────────────────────────────────────────────
    print("Loading EBM condition model...")
    encoder_ebm = ReactionEncoder(
        vocab_size=len(tokenizer.vocab), d_model=enc_cfg["d_model"],
        n_layers=enc_cfg["n_layers"], n_heads=enc_cfg["n_heads"],
        d_ff=enc_cfg["d_ff"], dropout=0, max_seq_len=enc_cfg["max_seq_len"],
    )
    ebm_head = ConditionManifoldEBM(
        reaction_dim=enc_cfg["d_model"],
        condition_dim=128,
        hidden_dims=[512, 256, 128],
        discrete_vocab_sizes=discrete_vocab_sizes,
        n_continuous=cond_encoder.n_continuous,
    )

    ebm_ckpt = ckpt_dir / "condition_ebm_random_seed42" / "best.pt"
    if ebm_ckpt.exists():
        ckpt = torch.load(ebm_ckpt, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"]
        enc_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
        head_state = {k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}
        encoder_ebm.load_state_dict(enc_state)
        ebm_head.load_state_dict(head_state)
        print(f"  Loaded from {ebm_ckpt}")
    else:
        print(f"  WARNING: No checkpoint at {ebm_ckpt}")

    encoder_ebm.to(device).eval()
    ebm_head.to(device).eval()

    # ─── Evaluate Point Model ────────────────────────────────────────
    print("\n--- Evaluating Point Model ---")
    point_correct_top1 = 0
    point_correct_top5 = 0
    point_marginal_correct = {name: 0 for name in discrete_vocab_sizes}
    n_total = 0

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            enc_out = encoder_point(batch["input_ids"], batch["attention_mask"])
            preds = point_head(enc_out["pooled"])

            targets = batch["discrete_conditions"]
            bs = targets.size(0)
            n_total += bs

            # Top-1 per component
            for i, name in enumerate(point_head.discrete_fields):
                logits = preds["discrete_logits"][name]
                pred_top1 = logits.argmax(dim=-1)
                point_marginal_correct[name] += (pred_top1 == targets[:, i]).sum().item()

                # Top-5
                if i == 0:  # check full tuple match only on first pass
                    pred_top5 = logits.topk(5, dim=-1).indices

    print(f"  Total test samples: {n_total}")
    print(f"  Marginal top-1 accuracy per component:")
    for name, correct in point_marginal_correct.items():
        acc = correct / max(n_total, 1) * 100
        print(f"    {name}: {acc:.1f}%")
    avg_marginal = np.mean([v / max(n_total, 1) for v in point_marginal_correct.values()]) * 100
    print(f"  Average marginal top-1: {avg_marginal:.1f}%")

    # ─── Evaluate EBM Model ──────────────────────────────────────────
    print("\n--- Evaluating EBM Model ---")
    pos_energies = []
    neg_energies = []
    nce_correct = 0
    nce_total = 0
    discrete_vocab_size_list = [len(v) for v in discrete_vocabs.values()]

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            enc_out = encoder_ebm(batch["input_ids"], batch["attention_mask"])
            rxn_repr = enc_out["pooled"]

            # Energy of true conditions
            e_pos = ebm_head.energy(rxn_repr, batch["discrete_conditions"], batch["continuous_conditions"])
            pos_energies.extend(e_pos.cpu().numpy().tolist())

            # Energy of random (negative) conditions
            neg_d, neg_c = generate_negative_samples(
                batch["discrete_conditions"], batch["continuous_conditions"],
                n_negatives=1, noise_scale=0.1, discrete_vocab_sizes=discrete_vocab_size_list,
            )
            neg_d = neg_d.to(device)
            neg_c = neg_c.to(device)
            e_neg = ebm_head.energy(rxn_repr, neg_d, neg_c)
            neg_energies.extend(e_neg.cpu().numpy().tolist())

            # NCE accuracy: positive should have lower energy
            nce_correct += (e_pos < e_neg).sum().item()
            nce_total += len(e_pos)

    pos_energies = np.array(pos_energies)
    neg_energies = np.array(neg_energies)

    energy_gap = neg_energies.mean() - pos_energies.mean()
    nce_acc = nce_correct / max(nce_total, 1) * 100

    # Coverage: what fraction of true conditions have energy below various thresholds?
    energy_thresholds = np.percentile(pos_energies, [50, 70, 90, 95])
    print(f"  Energy gap (neg - pos): {energy_gap:.4f}")
    print(f"  NCE accuracy: {nce_acc:.1f}%")
    print(f"  Positive energy: mean={pos_energies.mean():.4f}, std={pos_energies.std():.4f}")
    print(f"  Negative energy: mean={neg_energies.mean():.4f}, std={neg_energies.std():.4f}")

    # Coverage@alpha: fraction of true conditions within the alpha-percentile of the energy landscape
    print(f"\n  Coverage analysis:")
    for alpha in [0.5, 0.7, 0.9, 0.95]:
        threshold = np.percentile(np.concatenate([pos_energies, neg_energies]), alpha * 100)
        coverage = (pos_energies <= threshold).mean()
        print(f"    Coverage@{alpha}: {coverage:.3f} (threshold={threshold:.4f})")

    # ─── Comparison Table ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON: Point vs EBM")
    print("=" * 60)
    print(f"{'Metric':<35} {'Point':>12} {'EBM':>12}")
    print("-" * 60)
    print(f"{'Avg marginal top-1 accuracy':<35} {avg_marginal:>11.1f}% {'N/A':>12}")
    print(f"{'NCE accuracy (pos < neg energy)':<35} {'N/A':>12} {nce_acc:>11.1f}%")
    print(f"{'Energy gap (neg - pos)':<35} {'N/A':>12} {energy_gap:>12.4f}")
    print(f"{'Coverage@0.9':<35} {'N/A':>12} {(pos_energies <= np.percentile(np.concatenate([pos_energies, neg_energies]), 90)).mean():>12.3f}")
    print("=" * 60)

    # Save results
    results = {
        "point": {
            "marginal_accuracy": {name: correct / max(n_total, 1) for name, correct in point_marginal_correct.items()},
            "avg_marginal_top1": avg_marginal,
            "n_test": n_total,
        },
        "ebm": {
            "energy_gap": float(energy_gap),
            "nce_accuracy": nce_acc,
            "pos_energy_mean": float(pos_energies.mean()),
            "neg_energy_mean": float(neg_energies.mean()),
            "coverage_0.5": float((pos_energies <= np.percentile(np.concatenate([pos_energies, neg_energies]), 50)).mean()),
            "coverage_0.7": float((pos_energies <= np.percentile(np.concatenate([pos_energies, neg_energies]), 70)).mean()),
            "coverage_0.9": float((pos_energies <= np.percentile(np.concatenate([pos_energies, neg_energies]), 90)).mean()),
            "coverage_0.95": float((pos_energies <= np.percentile(np.concatenate([pos_energies, neg_energies]), 95)).mean()),
        },
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "condition_point_vs_ebm.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_dir / 'condition_point_vs_ebm.json'}")


if __name__ == "__main__":
    main()
