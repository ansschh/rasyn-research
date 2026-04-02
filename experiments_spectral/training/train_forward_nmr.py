"""
Train Forward NMR Model (Experiment 1).

Structure → per-atom chemical shifts with calibrated uncertainty.
Uses synthetic NMR data for pipeline validation, then experimental data when available.
"""

import sys
import json
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import mol_to_graph, load_records
from models.mol_encoder import MolecularGraphEncoder
from models.forward_nmr import ForwardNMRModel


def load_records(path):
    import pandas as pd
    return pd.read_parquet(path)


class NMRDataset(Dataset):
    """Dataset for forward NMR prediction."""

    def __init__(self, mol_df, nmr_df, nucleus="13C", max_atoms=100):
        self.max_atoms = max_atoms
        self.nucleus = nucleus

        # Merge molecule and NMR data
        nmr_filtered = nmr_df[nmr_df["nucleus"] == nucleus].copy()
        self.data = nmr_filtered.merge(mol_df[["mol_id", "smiles"]], on="mol_id", how="inner", suffixes=("", "_mol"))

        # Use the correct smiles column
        if "smiles_mol" in self.data.columns:
            self.data["smiles"] = self.data["smiles_mol"]

        print(f"  NMR Dataset ({nucleus}): {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = row["smiles"]

        # Get molecular graph
        graph = mol_to_graph(smiles)
        if graph is None:
            # Return dummy
            return self._dummy()

        n_atoms = min(graph["n_atoms"], self.max_atoms)

        # Parse shifts
        import ast
        shifts = ast.literal_eval(row["shifts"]) if isinstance(row["shifts"], str) else row["shifts"]

        # Create atom mask (which atoms have shift labels)
        atom_types = json.loads(row.get("atom_types", "[]")) if "atom_types" in row.index else []

        # For 13C: mask carbon atoms; for 1H: mask hydrogen-bearing atoms
        atom_mask = np.zeros(self.max_atoms, dtype=np.float32)
        shift_values = np.zeros(self.max_atoms, dtype=np.float32)

        shift_idx = 0
        for i in range(min(n_atoms, len(atom_types))):
            if self.nucleus == "13C" and atom_types[i] == 6:  # Carbon
                if shift_idx < len(shifts):
                    atom_mask[i] = 1.0
                    shift_values[i] = shifts[shift_idx]
                    shift_idx += 1
            elif self.nucleus == "1H" and atom_types[i] == 1:  # Hydrogen
                if shift_idx < len(shifts):
                    atom_mask[i] = 1.0
                    shift_values[i] = shifts[shift_idx]
                    shift_idx += 1

        # If we couldn't assign properly, just assign sequentially to first N atoms
        if shift_idx == 0 and len(shifts) > 0:
            for i, s in enumerate(shifts[:self.max_atoms]):
                atom_mask[i] = 1.0
                shift_values[i] = s

        # Pad graph features
        node_feats = np.zeros((self.max_atoms, graph["node_features"].shape[1]), dtype=np.float32)
        node_feats[:n_atoms] = graph["node_features"][:n_atoms]

        return {
            "node_features": torch.tensor(node_feats),
            "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long),
            "edge_features": torch.tensor(graph["edge_features"]),
            "atom_mask": torch.tensor(atom_mask),
            "shift_values": torch.tensor(shift_values),
            "n_atoms": n_atoms,
        }

    def _dummy(self):
        return {
            "node_features": torch.zeros(self.max_atoms, 13),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "edge_features": torch.zeros(0, 3),
            "atom_mask": torch.zeros(self.max_atoms),
            "shift_values": torch.zeros(self.max_atoms),
            "n_atoms": 0,
        }


def collate_nmr(batch):
    """Custom collation for variable-size graphs."""
    # For simplicity, pad to max_atoms (already done in dataset)
    return {
        "node_features": torch.stack([b["node_features"] for b in batch]),
        "edge_index": [b["edge_index"] for b in batch],  # keep as list (variable size)
        "edge_features": [b["edge_features"] for b in batch],
        "atom_mask": torch.stack([b["atom_mask"] for b in batch]),
        "shift_values": torch.stack([b["shift_values"] for b in batch]),
        "n_atoms": [b["n_atoms"] for b in batch],
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

    # Load data
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    mol_df = load_records(processed_dir / "molecules.parquet")
    nmr_df = load_records(processed_dir / "nmr_spectra.parquet")
    print(f"Loaded {len(mol_df)} molecules, {len(nmr_df)} NMR records")

    # Create dataset
    dataset = NMRDataset(mol_df, nmr_df, nucleus=args.nucleus)

    # Split
    n = len(dataset)
    n_val = max(1, n // 10)
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_nmr, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_nmr, num_workers=0)

    # Models
    mol_encoder = MolecularGraphEncoder(node_input_dim=13, edge_input_dim=3, hidden_dim=256, n_layers=4)
    nmr_model = ForwardNMRModel(atom_repr_dim=256, hidden_dims=[256, 128])

    mol_encoder.to(device)
    nmr_model.to(device)

    n_params = sum(p.numel() for p in mol_encoder.parameters()) + sum(p.numel() for p in nmr_model.parameters())
    print(f"Total parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        list(mol_encoder.parameters()) + list(nmr_model.parameters()),
        lr=args.lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_dir = Path(__file__).parent.parent / "checkpoints" / f"forward_nmr_{args.nucleus}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        # Train
        mol_encoder.train()
        nmr_model.train()
        train_loss = 0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Process each molecule in batch independently (variable graph sizes)
            batch_loss = torch.tensor(0.0, device=device)
            for i in range(len(batch["n_atoms"])):
                n = batch["n_atoms"][i]
                if n == 0:
                    continue

                node_f = batch["node_features"][i, :n].to(device)
                edge_idx = batch["edge_index"][i].to(device)
                edge_f = batch["edge_features"][i].to(device)
                mask = batch["atom_mask"][i, :n].to(device)
                shifts = batch["shift_values"][i, :n].to(device)

                if mask.sum() == 0:
                    continue

                enc_out = mol_encoder(node_f, edge_idx, edge_f)
                preds = nmr_model(enc_out["atom_repr"])
                losses = nmr_model.compute_loss(preds, shifts, mask)
                batch_loss = batch_loss + losses["total"]

            if batch_loss.requires_grad:
                batch_loss = batch_loss / max(len(batch["n_atoms"]), 1)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(mol_encoder.parameters()) + list(nmr_model.parameters()), 1.0
                )
                optimizer.step()
                train_loss += batch_loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validate
        mol_encoder.eval()
        nmr_model.eval()
        val_loss = 0
        val_mae = 0
        n_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch_loss = 0
                batch_mae = 0
                count = 0
                for i in range(len(batch["n_atoms"])):
                    n = batch["n_atoms"][i]
                    if n == 0:
                        continue

                    node_f = batch["node_features"][i, :n].to(device)
                    edge_idx = batch["edge_index"][i].to(device)
                    edge_f = batch["edge_features"][i].to(device)
                    mask = batch["atom_mask"][i, :n].to(device)
                    shifts = batch["shift_values"][i, :n].to(device)

                    if mask.sum() == 0:
                        continue

                    enc_out = mol_encoder(node_f, edge_idx, edge_f)
                    preds = nmr_model(enc_out["atom_repr"])
                    losses = nmr_model.compute_loss(preds, shifts, mask)
                    batch_loss += losses["total"].item()
                    batch_mae += losses["mae"].item()
                    count += 1

                if count > 0:
                    val_loss += batch_loss / count
                    val_mae += batch_mae / count
                n_val_batches += 1

        val_loss /= max(n_val_batches, 1)
        val_mae /= max(n_val_batches, 1)

        scheduler.step(val_loss)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_mae={val_mae:.2f} ppm")

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "mol_encoder": mol_encoder.state_dict(),
                "nmr_model": nmr_model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "val_mae": val_mae,
            }, ckpt_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Forward NMR training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
