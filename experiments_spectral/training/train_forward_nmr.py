"""
Train Forward NMR Model (Experiment 1).

Structure → per-atom chemical shifts with calibrated uncertainty.
Uses real nmrshiftdb2 data (57K molecules, 33K 13C spectra, 17K 1H spectra).
"""

import sys
import json
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import mol_to_graph
from models.mol_encoder import MolecularGraphEncoder
from models.forward_nmr import ForwardNMRModel


class NMRShiftDataset(Dataset):
    """
    Dataset for forward NMR prediction from real nmrshiftdb2 data.

    Each sample: (molecular_graph, atom_shifts, atom_mask)
    The nmr_spectra.parquet has: smiles, nucleus, shifts (JSON list), atom_indices (JSON list)
    """

    def __init__(self, df: pd.DataFrame, nucleus: str = "13C", max_atoms: int = 100):
        self.max_atoms = max_atoms
        self.nucleus = nucleus

        # Filter to target nucleus
        self.data = df[df["nucleus"] == nucleus].reset_index(drop=True)
        print(f"  NMR Dataset ({nucleus}): {len(self.data)} samples")

        # Pre-validate: only keep rows where we can parse the molecule
        valid_indices = []
        for i in range(len(self.data)):
            smi = self.data.iloc[i]["smiles"]
            graph = mol_to_graph(str(smi))
            if graph is not None and graph["n_atoms"] <= max_atoms:
                valid_indices.append(i)

            if i % 5000 == 0 and i > 0:
                print(f"    Validated {i}/{len(self.data)}, {len(valid_indices)} valid")

        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        print(f"  After validation: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = str(row["smiles"])

        graph = mol_to_graph(smiles)
        if graph is None:
            return self._dummy()

        n_atoms = min(graph["n_atoms"], self.max_atoms)

        # Parse shifts and atom indices
        shifts = json.loads(row["shifts"])
        atom_indices = json.loads(row["atom_indices"])

        # Create atom-level targets
        atom_mask = np.zeros(self.max_atoms, dtype=np.float32)
        shift_values = np.zeros(self.max_atoms, dtype=np.float32)

        for shift, aidx in zip(shifts, atom_indices):
            if 0 <= aidx < n_atoms:
                atom_mask[aidx] = 1.0
                shift_values[aidx] = shift

        # Pad node features
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
    return {
        "node_features": torch.stack([b["node_features"] for b in batch]),
        "edge_index": [b["edge_index"] for b in batch],
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

    # Load real NMR data
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    nmr_df = pd.read_parquet(processed_dir / "nmr_spectra.parquet")
    print(f"Loaded {len(nmr_df)} NMR records")

    # Create dataset (validation happens inside)
    dataset = NMRShiftDataset(nmr_df, nucleus=args.nucleus)

    if len(dataset) == 0:
        print("No valid samples found!")
        return

    # Split 80/10/10
    n = len(dataset)
    n_test = max(1, n // 10)
    n_val = max(1, n // 10)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

    print(f"Split: train={n_train}, val={n_val}, test={n_test}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_nmr, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_nmr, num_workers=0)

    # Models
    mol_encoder = MolecularGraphEncoder(node_input_dim=13, edge_input_dim=3, hidden_dim=256, n_layers=4).to(device)
    nmr_model = ForwardNMRModel(atom_repr_dim=256, hidden_dims=[256, 128]).to(device)

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
        train_losses, train_maes = [], []

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            count = 0

            for i in range(len(batch["n_atoms"])):
                n = batch["n_atoms"][i]
                if n == 0:
                    continue
                mask = batch["atom_mask"][i, :n].to(device)
                if mask.sum() == 0:
                    continue

                node_f = batch["node_features"][i, :n].to(device)
                edge_idx = batch["edge_index"][i].to(device)
                edge_f = batch["edge_features"][i].to(device)
                shifts = batch["shift_values"][i, :n].to(device)

                enc_out = mol_encoder(node_f, edge_idx, edge_f)
                preds = nmr_model(enc_out["atom_repr"])
                losses = nmr_model.compute_loss(preds, shifts, mask)
                batch_loss = batch_loss + losses["total"]
                count += 1

            if count > 0:
                avg_loss = batch_loss / count
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(mol_encoder.parameters()) + list(nmr_model.parameters()), 1.0
                )
                optimizer.step()
                train_losses.append(avg_loss.item())

        # Validate
        mol_encoder.eval()
        nmr_model.eval()
        val_losses, val_maes = [], []

        with torch.no_grad():
            for batch in val_loader:
                for i in range(len(batch["n_atoms"])):
                    n = batch["n_atoms"][i]
                    if n == 0:
                        continue
                    mask = batch["atom_mask"][i, :n].to(device)
                    if mask.sum() == 0:
                        continue

                    node_f = batch["node_features"][i, :n].to(device)
                    edge_idx = batch["edge_index"][i].to(device)
                    edge_f = batch["edge_features"][i].to(device)
                    shifts = batch["shift_values"][i, :n].to(device)

                    enc_out = mol_encoder(node_f, edge_idx, edge_f)
                    preds = nmr_model(enc_out["atom_repr"])
                    losses = nmr_model.compute_loss(preds, shifts, mask)
                    val_losses.append(losses["total"].item())
                    val_maes.append(losses["mae"].item())

        avg_train = np.mean(train_losses) if train_losses else 0
        avg_val = np.mean(val_losses) if val_losses else 0
        avg_mae = np.mean(val_maes) if val_maes else 0

        scheduler.step(avg_val)
        print(f"Epoch {epoch+1:3d}/{args.epochs} | train_loss={avg_train:.4f} | val_loss={avg_val:.4f} | val_mae={avg_mae:.2f} ppm")

        if avg_val < best_val_loss and val_losses:
            best_val_loss = avg_val
            patience_counter = 0
            torch.save({
                "mol_encoder": mol_encoder.state_dict(),
                "nmr_model": nmr_model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val,
                "val_mae": avg_mae,
            }, ckpt_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"Forward NMR training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
