"""
NMR Dataset for set-valued prediction.

Loads nmrshiftdb2 spectra and returns:
- Molecular graph (from SMILES, no atom index dependency)
- Set of true shifts (just the numbers, NO atom indices)
- Number of target atoms (carbons for 13C) in the molecule

The key difference from the per-atom dataset: we completely ignore atom indices.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from data.preprocessing import mol_to_graph

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# Map nucleus to target atomic number
NUCLEUS_TO_ATOMIC_NUM = {
    "13C": 6,
    "1H": 1,
    "19F": 9,
    "31P": 15,
    "15N": 7,
}


class SetNMRDataset(Dataset):
    """
    Dataset for set-valued NMR shift prediction.

    Returns per sample:
    - node_features, edge_index, edge_features: molecular graph
    - carbon_mask: which atoms are target element (C for 13C)
    - true_shifts: sorted list of ground truth shifts (variable length)
    - n_carbons: number of target atoms in the molecule
    - n_peaks: number of observed peaks
    """

    def __init__(
        self,
        nmr_df: pd.DataFrame,
        nucleus: str = "13C",
        max_atoms: int = 100,
        max_peaks: int = 50,
        prevalidate: bool = True,
    ):
        self.max_atoms = max_atoms
        self.max_peaks = max_peaks
        self.nucleus = nucleus
        self.target_atomic_num = NUCLEUS_TO_ATOMIC_NUM.get(nucleus, 6)

        # Filter to target nucleus
        data = nmr_df[nmr_df["nucleus"] == nucleus].reset_index(drop=True)
        print(f"  SetNMR Dataset ({nucleus}): {len(data)} raw samples")

        if prevalidate:
            valid = []
            for i in range(len(data)):
                row = data.iloc[i]
                smi = str(row["smiles"])

                # Validate molecule
                mol = Chem.MolFromSmiles(smi) if HAS_RDKIT else None
                if mol is None:
                    continue
                if mol.GetNumAtoms() > max_atoms:
                    continue

                # Count target atoms
                n_target = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == self.target_atomic_num)
                if n_target == 0:
                    continue

                # Parse shifts
                shifts = json.loads(row["shifts"])
                if len(shifts) == 0:
                    continue

                valid.append(i)

                if i % 5000 == 0 and i > 0:
                    print(f"    Validated {i}/{len(data)}, {len(valid)} valid")

            self.data = data.iloc[valid].reset_index(drop=True)
        else:
            self.data = data

        print(f"  After validation: {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smiles = str(row["smiles"])

        # Build molecular graph
        graph = mol_to_graph(smiles)
        if graph is None:
            return self._dummy()

        n_atoms = min(graph["n_atoms"], self.max_atoms)

        # Identify target atoms (e.g., carbons for 13C)
        mol = Chem.MolFromSmiles(smiles)
        carbon_mask = np.zeros(self.max_atoms, dtype=np.float32)
        for i in range(min(n_atoms, mol.GetNumAtoms())):
            if mol.GetAtomWithIdx(i).GetAtomicNum() == self.target_atomic_num:
                carbon_mask[i] = 1.0

        n_carbons = int(carbon_mask[:n_atoms].sum())

        # Parse true shifts — JUST the values, no atom indices
        shifts = json.loads(row["shifts"])
        shifts = sorted(shifts)[:self.max_peaks]  # sort for consistency
        n_peaks = len(shifts)

        # Pad shifts to max_peaks
        padded_shifts = np.zeros(self.max_peaks, dtype=np.float32)
        shift_mask = np.zeros(self.max_peaks, dtype=np.float32)
        for i, s in enumerate(shifts):
            padded_shifts[i] = s
            shift_mask[i] = 1.0

        # Pad node features
        node_feats = np.zeros((self.max_atoms, graph["node_features"].shape[1]), dtype=np.float32)
        node_feats[:n_atoms] = graph["node_features"][:n_atoms]

        return {
            "node_features": torch.tensor(node_feats),
            "edge_index": torch.tensor(graph["edge_index"], dtype=torch.long),
            "edge_features": torch.tensor(graph["edge_features"]),
            "carbon_mask": torch.tensor(carbon_mask),
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
            "true_shifts": torch.zeros(self.max_peaks),
            "shift_mask": torch.zeros(self.max_peaks),
            "n_atoms": 0,
            "n_carbons": 0,
            "n_peaks": 0,
        }


def collate_set_nmr(batch):
    """Custom collation for variable-size graphs."""
    return {
        "node_features": torch.stack([b["node_features"] for b in batch]),
        "edge_index": [b["edge_index"] for b in batch],
        "edge_features": [b["edge_features"] for b in batch],
        "carbon_mask": torch.stack([b["carbon_mask"] for b in batch]),
        "true_shifts": torch.stack([b["true_shifts"] for b in batch]),
        "shift_mask": torch.stack([b["shift_mask"] for b in batch]),
        "n_atoms": [b["n_atoms"] for b in batch],
        "n_carbons": [b["n_carbons"] for b in batch],
        "n_peaks": [b["n_peaks"] for b in batch],
    }
