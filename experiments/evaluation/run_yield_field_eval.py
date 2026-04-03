"""
Experiment 3 Evaluation: Yield Field Robustness Analysis.

Using the trained yield field ensemble on Doyle HTE:
- Standard metrics: RMSE, MAE, R2
- Feasibility: AUROC, Brier
- Uncertainty quality: calibration, error-uncertainty correlation
- Robustness: plateau widths, robustness-adjusted utility
- Key test: does the model prefer broad plateaus over narrow peaks?
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.featurization import SmilesTokenizer, ConditionEncoder, ReactionConditionDataset, create_dataloader
from data.preprocessing import load_records
from data.splits import random_split
from models.reaction_encoder import ReactionEncoder
from models.condition_manifold import ConditionEmbedder
from models.yield_field import YieldFeasibilityField, YieldFieldEnsemble
from training.utils import get_device
from evaluation.yield_metrics import (
    yield_regression_metrics, feasibility_metrics,
    uncertainty_quality, robustness_adjusted_utility,
)


def main():
    print("=" * 60)
    print("  Experiment 3: Yield Field Robustness Evaluation")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Load Doyle HTE data
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    df = load_records(processed_dir / "doyle_2018.parquet")
    df = df[df["yield_value"].notna()].reset_index(drop=True)
    print(f"Loaded {len(df)} Doyle HTE reactions with yields")

    # Load vocabs
    with open(processed_dir / "condition_vocab.json") as f:
        discrete_vocabs = json.load(f)
    with open(processed_dir / "continuous_stats.json") as f:
        continuous_stats = json.load(f)

    cond_encoder = ConditionEncoder(discrete_vocabs, continuous_stats)
    discrete_vocab_sizes = {k: len(v) for k, v in discrete_vocabs.items()}

    tokenizer_path = processed_dir / "smiles_vocab.json"
    if tokenizer_path.exists():
        tokenizer = SmilesTokenizer.load(tokenizer_path)
    else:
        from data.featurization import SmilesTokenizer
        tokenizer = SmilesTokenizer({"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3, "<MASK>": 4, "<SEP>": 5})

    # Use same split as training
    splits = random_split(df, test_frac=0.1, val_frac=0.1, seed=42)
    test_ds = ReactionConditionDataset(splits["test"], tokenizer, cond_encoder, max_seq_len=512)
    test_loader = create_dataloader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    enc_cfg = {"d_model": 256, "n_layers": 4, "n_heads": 4, "d_ff": 1024, "max_seq_len": 512}

    # ─── Load ensemble members ───────────────────────────────────────
    print("\nLoading yield field ensemble...")
    ckpt_dir = Path(__file__).parent.parent / "checkpoints"

    n_members = 5
    all_yield_preds = []
    all_feas_preds = []
    all_true_yields = []
    all_true_success = []
    loaded_members = 0

    for member_idx in range(n_members):
        member_dir = ckpt_dir / f"yield_field_doyle_2018_random_member{member_idx}"
        ckpt_path = member_dir / "best.pt"
        if not ckpt_path.exists():
            print(f"  Member {member_idx}: not found at {ckpt_path}")
            continue

        # Build models
        encoder = ReactionEncoder(
            vocab_size=max(len(tokenizer.vocab), 10),
            d_model=enc_cfg["d_model"], n_layers=enc_cfg["n_layers"],
            n_heads=enc_cfg["n_heads"], d_ff=enc_cfg["d_ff"],
            dropout=0, max_seq_len=enc_cfg["max_seq_len"],
        )
        cond_embedder = ConditionEmbedder(
            discrete_vocab_sizes=discrete_vocab_sizes,
            n_continuous=cond_encoder.n_continuous,
            embed_dim=32, output_dim=128,
        )
        yield_model = YieldFeasibilityField(
            reaction_dim=enc_cfg["d_model"], condition_dim=128,
            hidden_dims=[512, 256, 128], dropout=0,
        )

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"]

        enc_state = {k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}
        emb_state = {k.replace("cond_embedder.", ""): v for k, v in state.items() if k.startswith("cond_embedder.")}
        ym_state = {k.replace("yield_model.", ""): v for k, v in state.items() if k.startswith("yield_model.")}

        encoder.load_state_dict(enc_state)
        cond_embedder.load_state_dict(emb_state)
        yield_model.load_state_dict(ym_state)

        encoder.to(device).eval()
        cond_embedder.to(device).eval()
        yield_model.to(device).eval()

        print(f"  Member {member_idx}: loaded (val_loss={ckpt.get('metrics', {}).get('total', 'N/A')})")
        loaded_members += 1

        # Run inference on test set
        member_yields = []
        member_feas = []
        true_yields_collected = member_idx == 0  # only collect once

        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                enc_out = encoder(batch["input_ids"], batch["attention_mask"])
                cond_repr = cond_embedder(batch["discrete_conditions"], batch["continuous_conditions"])
                preds = yield_model(enc_out["pooled"], cond_repr)

                member_yields.append(preds["yield_mean"].cpu().numpy())
                member_feas.append(preds["feasibility_prob"].cpu().numpy())

                if member_idx == 0:
                    all_true_yields.append(batch["yield"].cpu().numpy())
                    all_true_success.append(batch["success"].cpu().numpy())

        all_yield_preds.append(np.concatenate(member_yields))
        all_feas_preds.append(np.concatenate(member_feas))

    if loaded_members == 0:
        print("No ensemble members found!")
        return

    # Stack predictions
    all_true_yields = np.concatenate(all_true_yields)
    all_true_success = np.concatenate(all_true_success)
    yield_preds = np.stack(all_yield_preds)  # (n_members, n_test)
    feas_preds = np.stack(all_feas_preds)

    # Ensemble mean and uncertainty
    ensemble_yield_mean = yield_preds.mean(axis=0)
    ensemble_yield_std = yield_preds.std(axis=0)  # epistemic uncertainty
    ensemble_feas_mean = feas_preds.mean(axis=0)
    ensemble_feas_std = feas_preds.std(axis=0)

    n_test = len(all_true_yields)
    print(f"\nEvaluating on {n_test} test samples, {loaded_members} ensemble members")

    # ─── Standard Metrics ─────────────────────────────────────────────
    print("\n--- Yield Regression ---")
    reg_metrics = yield_regression_metrics(all_true_yields, ensemble_yield_mean)
    for k, v in reg_metrics.items():
        print(f"  {k}: {v:.4f}")

    print("\n--- Feasibility Classification ---")
    feas_met = feasibility_metrics(all_true_success, ensemble_feas_mean)
    for k, v in feas_met.items():
        print(f"  {k}: {v:.4f}")

    # ─── Uncertainty Quality ──────────────────────────────────────────
    print("\n--- Uncertainty Quality ---")
    uncert = uncertainty_quality(all_true_yields, ensemble_yield_mean, ensemble_yield_std, ensemble_yield_std)
    for k, v in uncert.items():
        print(f"  {k}: {v:.4f}")

    # ─── Robustness Analysis ─────────────────────────────────────────
    print("\n--- Robustness-Adjusted Utility ---")
    # Simulate plateau width from ensemble disagreement
    # High ensemble agreement = robust prediction = wide plateau
    plateau_proxy = 1.0 - np.clip(ensemble_yield_std / 0.3, 0, 1)  # normalize
    robust_utility = robustness_adjusted_utility(ensemble_yield_mean, plateau_proxy, alpha=0.5)

    print(f"  Mean yield: {ensemble_yield_mean.mean():.4f}")
    print(f"  Mean plateau width (proxy): {plateau_proxy.mean():.4f}")
    print(f"  Mean robust utility: {robust_utility.mean():.4f}")

    # Do high-utility reactions actually have higher true yields?
    top_by_yield = np.argsort(-ensemble_yield_mean)[:50]
    top_by_robust = np.argsort(-robust_utility)[:50]

    true_yield_of_yield_top = all_true_yields[top_by_yield].mean()
    true_yield_of_robust_top = all_true_yields[top_by_robust].mean()

    print(f"\n  Top 50 by yield prediction: avg true yield = {true_yield_of_yield_top:.4f}")
    print(f"  Top 50 by robust utility:   avg true yield = {true_yield_of_robust_top:.4f}")

    # ─── Summary Table ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  YIELD FIELD EVALUATION SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<35} {'Value':>12}")
    print("-" * 48)
    for k, v in reg_metrics.items():
        print(f"  {k:<33} {v:>12.4f}")
    for k, v in feas_met.items():
        if k != "warning":
            print(f"  feas_{k:<29} {v:>12.4f}")
    print(f"  {'epistemic_error_corr':<33} {uncert.get('epistemic_error_corr', 0):>12.4f}")
    print(f"  {'within_1sigma':<33} {uncert.get('within_1sigma', 0):>12.4f}")
    print(f"  {'mean_robust_utility':<33} {robust_utility.mean():>12.4f}")
    print("=" * 60)

    # Save
    results = {
        "yield_regression": reg_metrics,
        "feasibility": feas_met,
        "uncertainty": uncert,
        "robustness": {
            "mean_yield": float(ensemble_yield_mean.mean()),
            "mean_plateau_width": float(plateau_proxy.mean()),
            "mean_robust_utility": float(robust_utility.mean()),
            "top50_yield_true": float(true_yield_of_yield_top),
            "top50_robust_true": float(true_yield_of_robust_top),
        },
        "n_test": n_test,
        "n_members": loaded_members,
    }

    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "yield_field_evaluation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {results_dir / 'yield_field_evaluation.json'}")


if __name__ == "__main__":
    main()


# Import needed at top
from data.featurization import SmilesTokenizer
