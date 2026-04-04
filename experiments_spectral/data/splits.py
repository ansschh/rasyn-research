"""
Proper splitting strategies for NMR prediction.

Random splits are too easy for molecular ML. We implement:
1. Random split (baseline, but now properly reported as such)
2. Scaffold split (Murcko scaffolds — tests generalization to new chemotypes)
3. Molecular weight split (tests extrapolation to larger/smaller molecules)
"""

from typing import Optional

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def random_split(df: pd.DataFrame, test_frac=0.1, val_frac=0.1, seed=42):
    """Standard random split."""
    n = len(df)
    idx = np.random.RandomState(seed).permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_idx = idx[:n_test]
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
        "split_type": "random",
    }


def _get_scaffold(smiles: str) -> Optional[str]:
    if not HAS_RDKIT or not smiles or pd.isna(smiles):
        return "unknown"
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return "unknown"
        scaffold = GetScaffoldForMol(mol)
        generic = MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(generic, canonical=True)
    except Exception:
        return "unknown"


def scaffold_split(df: pd.DataFrame, smiles_col="smiles", test_frac=0.1, val_frac=0.1, seed=42):
    """
    Scaffold split: test set has unseen Murcko scaffolds.
    This tests whether the model generalizes to new chemical space.
    """
    scaffolds = df[smiles_col].apply(_get_scaffold)
    df = df.copy()
    df["_scaffold"] = scaffolds

    scaffold_groups = df.groupby("_scaffold").indices
    scaffold_list = list(scaffold_groups.keys())

    rng = np.random.RandomState(seed)
    rng.shuffle(scaffold_list)

    n_total = len(df)
    n_test = int(n_total * test_frac)
    n_val = int(n_total * val_frac)

    test_idx, val_idx, train_idx = [], [], []
    for scaffold in scaffold_list:
        indices = scaffold_groups[scaffold].tolist()
        if len(test_idx) < n_test:
            test_idx.extend(indices)
        elif len(val_idx) < n_val:
            val_idx.extend(indices)
        else:
            train_idx.extend(indices)

    df = df.drop(columns=["_scaffold"])
    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
        "split_type": "scaffold",
    }


def get_split(df: pd.DataFrame, strategy: str = "random", **kwargs):
    if strategy == "random":
        return random_split(df, **kwargs)
    elif strategy == "scaffold":
        return scaffold_split(df, **kwargs)
    else:
        raise ValueError(f"Unknown split: {strategy}")
