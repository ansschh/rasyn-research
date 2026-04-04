"""
Train ALL baselines + SetNMR with proper experimental rigor.

Baselines:
1. per_atom_raw: per-atom prediction on raw nmrshiftdb2 indices (the broken baseline)
2. per_atom_carbon_only: per-atom on ONLY carbon-indexed shifts (cleaned)
3. per_atom_drop_invalid: drop all non-carbon assignments, per-atom on remainder
4. chamfer: Chamfer distance loss (simpler set loss, no Hungarian)
5. setnmr: Hungarian matching + spectrum reconstruction (our method)

Each baseline runs with:
- 3 seeds
- Both random and scaffold splits
- Reports TEST set MAE (not validation)
"""

import sys
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.preprocessing import mol_to_graph
from data.splits import get_split
from models.mol_encoder import MolecularGraphEncoder
from models.set_nmr import SetNMRPredictor, combined_set_nmr_loss

try:
    from rdkit import Chem
except ImportError:
    pass


# ─── Unified Dataset ─────────────────────────────────────────────────────────

class UnifiedNMRDataset(Dataset):
    """
    Dataset that supports all baseline modes.

    Modes:
    - "per_atom_raw": use raw atom indices as provided
    - "per_atom_carbon_only": only keep shifts assigned to carbon atoms
    - "set": predict set of shifts, no atom indices
    """

    def __init__(self, df, mode="set", nucleus="13C", max_atoms=100, max_peaks=50):
        self.max_atoms = max_atoms
        self.max_peaks = max_peaks
        self.mode = mode
        self.target_anum = 6 if nucleus == "13C" else 1

        data = df[df["nucleus"] == nucleus].reset_index(drop=True)

        # Pre-validate
        valid = []
        for i in range(len(data)):
            row = data.iloc[i]
            smi = str(row["smiles"])
            mol = Chem.MolFromSmiles(smi)
            if mol is None or mol.GetNumAtoms() > max_atoms:
                continue

            shifts = json.loads(row["shifts"])
            if len(shifts) == 0:
                continue

            if mode == "per_atom_carbon_only":
                # Filter to only carbon-assigned shifts
                atom_indices = json.loads(row["atom_indices"])
                clean_shifts = []
                clean_indices = []
                for s, aidx in zip(shifts, atom_indices):
                    if 0 <= aidx < mol.GetNumAtoms() and mol.GetAtomWithIdx(aidx).GetAtomicNum() == self.target_anum:
                        clean_shifts.append(s)
                        clean_indices.append(aidx)
                if len(clean_shifts) == 0:
                    continue

            n_target = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == self.target_anum)
            if n_target == 0:
                continue

            valid.append(i)

        self.data = data.iloc[valid].reset_index(drop=True)
        print(f"  Dataset ({mode}): {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = str(row["smiles"])
        graph = mol_to_graph(smiles)
        if graph is None:
            return self._dummy()

        mol = Chem.MolFromSmiles(smiles)
        n_atoms = min(graph["n_atoms"], self.max_atoms)

        # Node features
        node_feats = np.zeros((self.max_atoms, graph["node_features"].shape[1]), dtype=np.float32)
        node_feats[:n_atoms] = graph["node_features"][:n_atoms]

        # Carbon mask
        carbon_mask = np.zeros(self.max_atoms, dtype=np.float32)
        for i in range(min(n_atoms, mol.GetNumAtoms())):
            if mol.GetAtomWithIdx(i).GetAtomicNum() == self.target_anum:
                carbon_mask[i] = 1.0
        n_carbons = int(carbon_mask[:n_atoms].sum())

        shifts = json.loads(row["shifts"])
        atom_indices = json.loads(row["atom_indices"])

        if self.mode == "per_atom_raw":
            # Raw: assign shifts to provided atom indices
            atom_shifts = np.zeros(self.max_atoms, dtype=np.float32)
            atom_mask = np.zeros(self.max_atoms, dtype=np.float32)
            for s, aidx in zip(shifts, atom_indices):
                if 0 <= aidx < n_atoms:
                    atom_shifts[aidx] = s
                    atom_mask[aidx] = 1.0

            return {
                "node_features": torch.tensor(node_feats),
                "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long),
                "edge_features": torch.tensor(graph["edge_features"]),
                "carbon_mask": torch.tensor(carbon_mask),
                "atom_shifts": torch.tensor(atom_shifts),
                "atom_mask": torch.tensor(atom_mask),
                "true_shifts": torch.zeros(self.max_peaks),  # unused
                "shift_mask": torch.zeros(self.max_peaks),
                "n_atoms": n_atoms,
                "n_carbons": n_carbons,
                "n_peaks": 0,
            }

        elif self.mode == "per_atom_carbon_only":
            # Cleaned: only use carbon-assigned shifts
            atom_shifts = np.zeros(self.max_atoms, dtype=np.float32)
            atom_mask = np.zeros(self.max_atoms, dtype=np.float32)
            for s, aidx in zip(shifts, atom_indices):
                if 0 <= aidx < n_atoms:
                    if mol.GetAtomWithIdx(aidx).GetAtomicNum() == self.target_anum:
                        atom_shifts[aidx] = s
                        atom_mask[aidx] = 1.0

            return {
                "node_features": torch.tensor(node_feats),
                "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long),
                "edge_features": torch.tensor(graph["edge_features"]),
                "carbon_mask": torch.tensor(carbon_mask),
                "atom_shifts": torch.tensor(atom_shifts),
                "atom_mask": torch.tensor(atom_mask),
                "true_shifts": torch.zeros(self.max_peaks),
                "shift_mask": torch.zeros(self.max_peaks),
                "n_atoms": n_atoms,
                "n_carbons": n_carbons,
                "n_peaks": 0,
            }

        else:  # "set" mode
            shifts_sorted = sorted(shifts)[:self.max_peaks]
            n_peaks = len(shifts_sorted)
            padded_shifts = np.zeros(self.max_peaks, dtype=np.float32)
            shift_mask = np.zeros(self.max_peaks, dtype=np.float32)
            for i, s in enumerate(shifts_sorted):
                padded_shifts[i] = s
                shift_mask[i] = 1.0

            return {
                "node_features": torch.tensor(node_feats),
                "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long),
                "edge_features": torch.tensor(graph["edge_features"]),
                "carbon_mask": torch.tensor(carbon_mask),
                "atom_shifts": torch.zeros(self.max_atoms),
                "atom_mask": torch.zeros(self.max_atoms),
                "true_shifts": torch.tensor(padded_shifts),
                "shift_mask": torch.tensor(shift_mask),
                "n_atoms": n_atoms,
                "n_carbons": n_carbons,
                "n_peaks": n_peaks,
            }

    def _dummy(self):
        return {
            "node_features": torch.zeros(self.max_atoms, 13),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "edge_features": torch.zeros(0, 3),
            "carbon_mask": torch.zeros(self.max_atoms),
            "atom_shifts": torch.zeros(self.max_atoms),
            "atom_mask": torch.zeros(self.max_atoms),
            "true_shifts": torch.zeros(self.max_peaks),
            "shift_mask": torch.zeros(self.max_peaks),
            "n_atoms": 0, "n_carbons": 0, "n_peaks": 0,
        }


def collate_fn(batch):
    return {
        "node_features": torch.stack([b["node_features"] for b in batch]),
        "edge_index": [b["edge_index"] for b in batch],
        "edge_features": [b["edge_features"] for b in batch],
        "carbon_mask": torch.stack([b["carbon_mask"] for b in batch]),
        "atom_shifts": torch.stack([b["atom_shifts"] for b in batch]),
        "atom_mask": torch.stack([b["atom_mask"] for b in batch]),
        "true_shifts": torch.stack([b["true_shifts"] for b in batch]),
        "shift_mask": torch.stack([b["shift_mask"] for b in batch]),
        "n_atoms": [b["n_atoms"] for b in batch],
        "n_carbons": [b["n_carbons"] for b in batch],
        "n_peaks": [b["n_peaks"] for b in batch],
    }


# ─── Loss functions ───────────────────────────────────────────────────────────

def per_atom_loss(pred_shifts, atom_shifts, atom_mask):
    """Standard per-atom MSE loss."""
    if atom_mask.sum() == 0:
        return torch.tensor(0.0, device=pred_shifts.device), 0.0
    diff = (pred_shifts - atom_shifts) * atom_mask
    mse = (diff ** 2).sum() / atom_mask.sum()
    mae = diff.abs().sum().item() / atom_mask.sum().item()
    return mse, mae


def chamfer_loss(pred_shifts, true_shifts):
    """Chamfer distance: each pred finds nearest true, and vice versa."""
    if len(pred_shifts) == 0 or len(true_shifts) == 0:
        return torch.tensor(50.0, device=pred_shifts.device if len(pred_shifts) > 0 else true_shifts.device), 50.0

    # pred -> nearest true
    dists_p2t = (pred_shifts.unsqueeze(1) - true_shifts.unsqueeze(0)).abs()
    min_p2t = dists_p2t.min(dim=1).values.mean()

    # true -> nearest pred
    min_t2p = dists_p2t.min(dim=0).values.mean()

    loss = min_p2t + min_t2p
    mae = (min_p2t.item() + min_t2p.item()) / 2
    return loss, mae


# ─── Training loop ────────────────────────────────────────────────────────────

def train_and_evaluate(
    method: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 42,
    max_epochs: int = 100,
    patience: int = 15,
    batch_size: int = 32,
    lr: float = 3e-4,
):
    """Train one model and return test MAE."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset mode
    if method in ("per_atom_raw", "per_atom_carbon_only"):
        mode = method
    else:
        mode = "set"

    train_ds = UnifiedNMRDataset(train_df, mode=mode)
    val_ds = UnifiedNMRDataset(val_df, mode=mode)
    test_ds = UnifiedNMRDataset(test_df, mode=mode)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Model
    encoder = MolecularGraphEncoder(node_input_dim=13, edge_input_dim=3, hidden_dim=256, n_layers=4).to(device)
    predictor = SetNMRPredictor(atom_repr_dim=256, hidden_dims=[256, 128]).to(device)

    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=lr, weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(max_epochs):
        # Train
        encoder.train()
        predictor.train()
        train_maes = []

        for batch in train_loader:
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            count = 0

            for i in range(len(batch["n_atoms"])):
                n = batch["n_atoms"][i]
                nc = batch["n_carbons"][i]
                if n == 0 or nc == 0:
                    continue

                node_f = batch["node_features"][i, :n].to(device)
                edge_idx = batch["edge_index"][i].to(device)
                edge_f = batch["edge_features"][i].to(device)
                c_mask = batch["carbon_mask"][i, :n].to(device)

                enc_out = encoder(node_f, edge_idx, edge_f)

                if method in ("per_atom_raw", "per_atom_carbon_only"):
                    # Per-atom: predict for all atoms, loss on masked
                    preds = predictor(enc_out["atom_repr"])
                    a_shifts = batch["atom_shifts"][i, :n].to(device)
                    a_mask = batch["atom_mask"][i, :n].to(device)
                    if a_mask.sum() == 0:
                        continue
                    loss, mae = per_atom_loss(preds["shift_mean"], a_shifts, a_mask)

                elif method == "chamfer":
                    # Chamfer on carbon atoms
                    carbon_idx = torch.where(c_mask > 0)[0]
                    if len(carbon_idx) == 0:
                        continue
                    carbon_repr = enc_out["atom_repr"][carbon_idx]
                    preds = predictor(carbon_repr)
                    np_peaks = batch["n_peaks"][i]
                    if np_peaks == 0:
                        continue
                    true_s = batch["true_shifts"][i, :np_peaks].to(device)
                    loss, mae = chamfer_loss(preds["shift_mean"], true_s)

                else:  # setnmr
                    carbon_idx = torch.where(c_mask > 0)[0]
                    if len(carbon_idx) == 0:
                        continue
                    carbon_repr = enc_out["atom_repr"][carbon_idx]
                    preds = predictor(carbon_repr)
                    np_peaks = batch["n_peaks"][i]
                    if np_peaks == 0:
                        continue
                    true_s = batch["true_shifts"][i, :np_peaks].to(device)
                    losses = combined_set_nmr_loss(preds["shift_mean"], true_s, preds["shift_logvar"])
                    loss = losses["total"]
                    mae = losses["matched_mae"].item()

                batch_loss = batch_loss + loss
                train_maes.append(mae)
                count += 1

            if count > 0:
                (batch_loss / count).backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(predictor.parameters()), 1.0
                )
                optimizer.step()

        # Validate
        val_mae = _evaluate(encoder, predictor, val_loader, method, device)
        scheduler.step(val_mae)

        train_mae_avg = np.mean(train_maes) if train_maes else 0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d} | train_mae={train_mae_avg:.2f} | val_mae={val_mae:.2f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            best_state = {
                "encoder": {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                "predictor": {k: v.cpu().clone() for k, v in predictor.state_dict().items()},
            }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

    # Load best and evaluate on TEST set
    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        predictor.load_state_dict(best_state["predictor"])
    encoder.to(device)
    predictor.to(device)

    test_mae = _evaluate(encoder, predictor, test_loader, method, device)
    print(f"  FINAL TEST MAE: {test_mae:.2f} ppm (val: {best_val_mae:.2f})")

    return {
        "test_mae": test_mae,
        "val_mae": best_val_mae,
        "method": method,
        "seed": seed,
    }


@torch.no_grad()
def _evaluate(encoder, predictor, loader, method, device):
    """Compute MAE on a dataset using Hungarian matching for all methods."""
    encoder.eval()
    predictor.eval()
    all_maes = []

    for batch in loader:
        for i in range(len(batch["n_atoms"])):
            n = batch["n_atoms"][i]
            nc = batch["n_carbons"][i]
            if n == 0 or nc == 0:
                continue

            node_f = batch["node_features"][i, :n].to(device)
            edge_idx = batch["edge_index"][i].to(device)
            edge_f = batch["edge_features"][i].to(device)
            c_mask = batch["carbon_mask"][i, :n].to(device)

            enc_out = encoder(node_f, edge_idx, edge_f)

            if method in ("per_atom_raw", "per_atom_carbon_only"):
                preds = predictor(enc_out["atom_repr"])
                pred_shifts = preds["shift_mean"][c_mask > 0].cpu().numpy()
            else:
                carbon_idx = torch.where(c_mask > 0)[0]
                if len(carbon_idx) == 0:
                    continue
                preds = predictor(enc_out["atom_repr"][carbon_idx])
                pred_shifts = preds["shift_mean"].cpu().numpy()

            # Get true shifts for Hungarian eval (fair comparison)
            np_peaks = batch["n_peaks"][i]
            if np_peaks > 0:
                true_shifts = batch["true_shifts"][i, :np_peaks].numpy()
            else:
                # For per-atom modes, reconstruct from atom_shifts
                a_mask = batch["atom_mask"][i, :n].numpy()
                a_shifts = batch["atom_shifts"][i, :n].numpy()
                true_shifts = a_shifts[a_mask > 0]

            if len(pred_shifts) == 0 or len(true_shifts) == 0:
                continue

            # Hungarian matching for fair MAE comparison across ALL methods
            n_p, n_t = len(pred_shifts), len(true_shifts)
            n_max = max(n_p, n_t)
            cost = np.full((n_max, n_max), 50.0)
            cost[:n_p, :n_t] = np.abs(pred_shifts[:, None] - true_shifts[None, :])
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_errors = []
            for r, c in zip(row_ind, col_ind):
                if r < n_p and c < n_t:
                    matched_errors.append(abs(pred_shifts[r] - true_shifts[c]))

            if matched_errors:
                all_maes.append(np.mean(matched_errors))

    return np.mean(all_maes) if all_maes else 999.0


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+",
                        default=["per_atom_raw", "per_atom_carbon_only", "chamfer", "setnmr"])
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    nmr_df = pd.read_parquet(processed_dir / "nmr_spectra.parquet")
    print(f"Loaded {len(nmr_df)} NMR records")

    all_results = []

    for split_type in args.splits:
        print(f"\n{'='*60}")
        print(f"  Split: {split_type}")
        print(f"{'='*60}")

        splits = get_split(nmr_df, strategy=split_type)
        print(f"  Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")

        for method in args.methods:
            for seed in args.seeds:
                print(f"\n--- {method} | seed={seed} ---")
                t0 = time.time()
                result = train_and_evaluate(
                    method=method,
                    train_df=splits["train"],
                    val_df=splits["val"],
                    test_df=splits["test"],
                    seed=seed,
                    max_epochs=args.epochs,
                    batch_size=args.batch_size,
                )
                result["split"] = split_type
                result["elapsed_s"] = round(time.time() - t0, 1)
                all_results.append(result)

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Method':<25} {'Split':<10} {'Test MAE (mean +/- std)':>25}")
    print(f"{'-'*80}")

    for split_type in args.splits:
        for method in args.methods:
            maes = [r["test_mae"] for r in all_results
                    if r["method"] == method and r["split"] == split_type]
            if maes:
                mean_mae = np.mean(maes)
                std_mae = np.std(maes)
                print(f"{method:<25} {split_type:<10} {mean_mae:>10.2f} +/- {std_mae:.2f} ppm")

    print(f"{'='*80}")

    # Save
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "all_baselines.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {results_dir / 'all_baselines.json'}")


if __name__ == "__main__":
    main()
