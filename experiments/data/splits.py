"""
Multiple split strategies for rigorous evaluation.

The proposal emphasizes that random splits hide important failure modes.
We implement:
1. Random split (baseline)
2. Scaffold split (OOD generalization to new substrates)
3. Source split (cross-lab/source transfer)
4. Time split (temporal generalization)
5. Perturbation split (robustness to condition shifts)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def random_split(
    df: pd.DataFrame,
    test_frac: float = 0.1,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Standard random split."""
    train_val, test = train_test_split(df, test_size=test_frac, random_state=seed)
    val_size = val_frac / (1 - test_frac)
    train, val = train_test_split(train_val, test_size=val_size, random_state=seed)

    return {"train": train.reset_index(drop=True),
            "val": val.reset_index(drop=True),
            "test": test.reset_index(drop=True)}


def _get_scaffold(smiles: str) -> Optional[str]:
    """Get Murcko scaffold for a SMILES string."""
    if not HAS_RDKIT or not smiles or pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaffold = GetScaffoldForMol(mol)
        generic = MakeScaffoldGeneric(scaffold)
        return Chem.MolToSmiles(generic, canonical=True)
    except Exception:
        return None


def scaffold_split(
    df: pd.DataFrame,
    smiles_col: str = "product_smiles",
    test_frac: float = 0.1,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """
    Split by Murcko scaffold so test set has unseen scaffolds.
    This tests OOD generalization to new chemical space.
    """
    if not HAS_RDKIT:
        print("WARNING: RDKit not available, falling back to random split")
        return random_split(df, test_frac, val_frac, seed)

    # Compute scaffolds
    scaffolds = df[smiles_col].apply(_get_scaffold)
    df = df.copy()
    df["_scaffold"] = scaffolds

    # Group by scaffold
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
    }


def source_split(
    df: pd.DataFrame,
    source_col: str = "source",
    test_sources: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Split by data source to test cross-lab/source transfer.
    Train on some sources, test on held-out sources.
    """
    sources = df[source_col].unique()
    if test_sources is None:
        # Hold out ~20% of sources
        n_test = max(1, len(sources) // 5)
        test_sources = list(sources[:n_test])

    test_mask = df[source_col].isin(test_sources)
    test = df[test_mask].reset_index(drop=True)
    train_val = df[~test_mask].reset_index(drop=True)

    # Split remaining into train/val
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)

    return {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
        "test_sources": test_sources,
    }


def condition_perturbation_split(
    df: pd.DataFrame,
    seed: int = 42,
    test_frac: float = 0.1,
    val_frac: float = 0.1,
    perturbation_configs: Optional[list[dict]] = None,
) -> dict:
    """
    Standard random split + generate perturbed versions of test conditions.
    This directly measures robustness to condition shifts.
    """
    splits = random_split(df, test_frac, val_frac, seed)

    if perturbation_configs is None:
        perturbation_configs = [
            {"name": "temp_+10", "col": "temperature", "delta": 10.0},
            {"name": "temp_-10", "col": "temperature", "delta": -10.0},
            {"name": "temp_+20", "col": "temperature", "delta": 20.0},
        ]

    # Generate perturbed test sets
    perturbed_tests = {}
    for config in perturbation_configs:
        test_copy = splits["test"].copy()
        col = config["col"]
        if col in test_copy.columns:
            mask = test_copy[col].notna()
            test_copy.loc[mask, col] = test_copy.loc[mask, col] + config["delta"]
        perturbed_tests[config["name"]] = test_copy

    splits["perturbed_tests"] = perturbed_tests
    return splits


# ─── Convenience ──────────────────────────────────────────────────────────────

SPLIT_REGISTRY = {
    "random": random_split,
    "scaffold": scaffold_split,
    "source": source_split,
    "perturbation": condition_perturbation_split,
}


def get_splits(
    df: pd.DataFrame,
    strategy: str = "random",
    **kwargs,
) -> dict:
    """Get train/val/test splits using the specified strategy."""
    if strategy not in SPLIT_REGISTRY:
        raise ValueError(f"Unknown split strategy: {strategy}. Options: {list(SPLIT_REGISTRY.keys())}")
    return SPLIT_REGISTRY[strategy](df, **kwargs)
