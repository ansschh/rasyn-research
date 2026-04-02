"""
Train SetNMR — Set-valued NMR prediction with Hungarian matching.

Uses the new architecture that sidesteps atom index alignment issues
by predicting the SET of shifts and using optimal assignment for the loss.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.nmr_dataset import SetNMRDataset, collate_set_nmr
from models.mol_encoder import MolecularGraphEncoder
from models.set_nmr import SetNMRPredictor, combined_set_nmr_loss


def train_one_epoch(mol_encoder, nmr_predictor, train_loader, optimizer, device):
    """Train for one epoch."""
    mol_encoder.train()
    nmr_predictor.train()

    epoch_losses = []
    epoch_maes = []

    for batch in train_loader:
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        batch_maes = []
        count = 0

        for i in range(len(batch["n_atoms"])):
            n = batch["n_atoms"][i]
            n_carbons = batch["n_carbons"][i]
            n_peaks = batch["n_peaks"][i]

            if n == 0 or n_carbons == 0 or n_peaks == 0:
                continue

            # Encode molecule
            node_f = batch["node_features"][i, :n].to(device)
            edge_idx = batch["edge_index"][i].to(device)
            edge_f = batch["edge_features"][i].to(device)
            c_mask = batch["carbon_mask"][i, :n].to(device)

            enc_out = mol_encoder(node_f, edge_idx, edge_f)
            atom_repr = enc_out["atom_repr"]  # (n_atoms, hidden_dim)

            # Extract carbon atom representations
            carbon_indices = torch.where(c_mask > 0)[0]
            if len(carbon_indices) == 0:
                continue
            carbon_repr = atom_repr[carbon_indices]  # (n_carbons, hidden_dim)

            # Predict shifts for each carbon
            preds = nmr_predictor(carbon_repr)
            pred_shifts = preds["shift_mean"]       # (n_carbons,)
            pred_logvar = preds["shift_logvar"]      # (n_carbons,)

            # True shifts (no atom indices needed!)
            true_shifts = batch["true_shifts"][i, :n_peaks].to(device)

            # Combined loss: Hungarian matching + spectrum reconstruction
            losses = combined_set_nmr_loss(
                pred_shifts, true_shifts, pred_logvar,
                hungarian_weight=1.0,
                spectrum_weight=0.5,
            )

            batch_loss = batch_loss + losses["total"]
            batch_maes.append(losses["matched_mae"].item())
            count += 1

        if count > 0:
            avg_loss = batch_loss / count
            avg_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(mol_encoder.parameters()) + list(nmr_predictor.parameters()), 1.0
            )
            optimizer.step()
            epoch_losses.append(avg_loss.item())
            epoch_maes.extend(batch_maes)

    return {
        "loss": np.mean(epoch_losses) if epoch_losses else 0,
        "mae": np.mean(epoch_maes) if epoch_maes else 0,
    }


@torch.no_grad()
def validate(mol_encoder, nmr_predictor, val_loader, device):
    """Validate."""
    mol_encoder.eval()
    nmr_predictor.eval()

    val_losses = []
    val_maes = []
    val_spec_losses = []

    for batch in val_loader:
        for i in range(len(batch["n_atoms"])):
            n = batch["n_atoms"][i]
            n_carbons = batch["n_carbons"][i]
            n_peaks = batch["n_peaks"][i]

            if n == 0 or n_carbons == 0 or n_peaks == 0:
                continue

            node_f = batch["node_features"][i, :n].to(device)
            edge_idx = batch["edge_index"][i].to(device)
            edge_f = batch["edge_features"][i].to(device)
            c_mask = batch["carbon_mask"][i, :n].to(device)

            enc_out = mol_encoder(node_f, edge_idx, edge_f)
            carbon_indices = torch.where(c_mask > 0)[0]
            if len(carbon_indices) == 0:
                continue
            carbon_repr = enc_out["atom_repr"][carbon_indices]

            preds = nmr_predictor(carbon_repr)
            true_shifts = batch["true_shifts"][i, :n_peaks].to(device)

            losses = combined_set_nmr_loss(
                preds["shift_mean"], true_shifts, preds["shift_logvar"],
            )

            val_losses.append(losses["total"].item())
            val_maes.append(losses["matched_mae"].item())
            val_spec_losses.append(losses["spectrum_loss"].item())

    return {
        "loss": np.mean(val_losses) if val_losses else 0,
        "mae": np.mean(val_maes) if val_maes else 0,
        "spec_loss": np.mean(val_spec_losses) if val_spec_losses else 0,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nucleus", choices=["1H", "13C"], default="13C")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load real NMR data
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    nmr_df = pd.read_parquet(processed_dir / "nmr_spectra.parquet")
    print(f"Loaded {len(nmr_df)} NMR records")

    # Create dataset (ignores atom indices — only uses shift values)
    dataset = SetNMRDataset(nmr_df, nucleus=args.nucleus)

    if len(dataset) == 0:
        print("No valid samples!")
        return

    # Split 80/10/10
    n = len(dataset)
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])
    print(f"Split: train={n_train}, val={n_val}, test={n_test}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_set_nmr, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_set_nmr, num_workers=0)

    # Models
    mol_encoder = MolecularGraphEncoder(node_input_dim=13, edge_input_dim=3, hidden_dim=256, n_layers=4).to(device)
    nmr_predictor = SetNMRPredictor(atom_repr_dim=256, hidden_dims=[256, 128]).to(device)

    n_params = sum(p.numel() for p in mol_encoder.parameters()) + sum(p.numel() for p in nmr_predictor.parameters())
    print(f"Total parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        list(mol_encoder.parameters()) + list(nmr_predictor.parameters()),
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_mae = float("inf")
    patience_counter = 0
    ckpt_dir = Path(__file__).parent.parent / "checkpoints" / f"set_nmr_{args.nucleus}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(mol_encoder, nmr_predictor, train_loader, optimizer, device)
        val_metrics = validate(mol_encoder, nmr_predictor, val_loader, device)

        scheduler.step(val_metrics["loss"])
        lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train_loss={train_metrics['loss']:.4f} mae={train_metrics['mae']:.2f} | "
              f"val_loss={val_metrics['loss']:.4f} mae={val_metrics['mae']:.2f} spec={val_metrics['spec_loss']:.4f} | "
              f"lr={lr:.1e}")

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            patience_counter = 0
            torch.save({
                "mol_encoder": mol_encoder.state_dict(),
                "nmr_predictor": nmr_predictor.state_dict(),
                "epoch": epoch,
                "val_mae": val_metrics["mae"],
                "val_loss": val_metrics["loss"],
            }, ckpt_dir / "best.pt")
            print(f"  -> New best MAE: {best_val_mae:.2f} ppm")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nSetNMR training complete.")
    print(f"Best val MAE: {best_val_mae:.2f} ppm (vs 35 ppm per-atom baseline)")


if __name__ == "__main__":
    main()
