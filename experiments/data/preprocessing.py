"""
Unified preprocessing for reaction data across ORDerly and Buchwald HTE sources.

Produces a standardized ReactionRecord format:
- reaction SMILES (product >> reactants)
- discrete conditions (solvent, catalyst, reagent, base)
- continuous conditions (temperature, time, concentration)
- outcome (yield, success/failure)
- metadata (source, dataset)
"""

import re
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available. Some features disabled.")


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class ReactionRecord:
    """Unified reaction representation."""
    reaction_id: str
    product_smiles: str
    reactant_smiles: str           # dot-separated if multiple
    reaction_smiles: str           # product>>reactants

    # Discrete conditions
    solvent: Optional[str] = None
    catalyst: Optional[str] = None
    reagent: Optional[str] = None
    base: Optional[str] = None
    ligand: Optional[str] = None

    # Continuous conditions
    temperature: Optional[float] = None    # Celsius
    time_hours: Optional[float] = None
    concentration: Optional[float] = None  # mol/L

    # Outcomes
    yield_value: Optional[float] = None    # 0-100
    success: Optional[bool] = None         # yield > threshold

    # Metadata
    source: str = ""                       # "orderly", "buchwald", etc.
    dataset: str = ""                      # specific dataset name


def canonicalize_smiles(smi: str) -> Optional[str]:
    """Canonicalize a SMILES string using RDKit."""
    if not HAS_RDKIT or not smi or pd.isna(smi):
        return smi
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        # Strip atom mapping
        for atom in mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber") if atom.HasProp("molAtomMapNumber") else None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# ─── ORDerly preprocessing ───────────────────────────────────────────────────

def preprocess_orderly(data_dir: Path, success_threshold: float = 5.0) -> list[ReactionRecord]:
    """
    Preprocess ORDerly condition dataset.

    ORDerly provides cleaned reaction data with:
    - reaction SMILES
    - agent_1, agent_2 (solvents/reagents)
    - catalyst_1, catalyst_2
    - product_yield
    - temperature
    """
    csv_path = data_dir / "orderly" / "orderly_condition_data.csv.gz"
    if not csv_path.exists():
        print(f"ORDerly data not found at {csv_path}")
        print("Run download.py first.")
        return []

    print(f"Loading ORDerly from {csv_path}...")
    df = pd.read_csv(csv_path, compression="gzip")
    print(f"  Loaded {len(df)} reactions")

    records = []
    for idx, row in df.iterrows():
        rxn_smi = row.get("rxn_smiles", row.get("canonical_rxn", ""))
        if not rxn_smi or pd.isna(rxn_smi):
            continue

        # Parse reaction SMILES: reactants>>products
        parts = str(rxn_smi).split(">>")
        if len(parts) != 2:
            continue

        reactants_smi, products_smi = parts[0], parts[1]
        product_can = canonicalize_smiles(products_smi)
        reactant_can = canonicalize_smiles(reactants_smi)

        if product_can is None or reactant_can is None:
            continue

        yield_val = row.get("product_yield", row.get("yield", None))
        if yield_val is not None and not pd.isna(yield_val):
            yield_val = float(yield_val)
        else:
            yield_val = None

        temp = row.get("temperature", None)
        if temp is not None and not pd.isna(temp):
            temp = float(temp)
        else:
            temp = None

        rec = ReactionRecord(
            reaction_id=f"orderly_{idx}",
            product_smiles=product_can,
            reactant_smiles=reactant_can,
            reaction_smiles=f"{product_can}>>{reactant_can}",
            solvent=str(row.get("agent_1", "")) if not pd.isna(row.get("agent_1", np.nan)) else None,
            catalyst=str(row.get("catalyst_1", "")) if not pd.isna(row.get("catalyst_1", np.nan)) else None,
            reagent=str(row.get("agent_2", "")) if not pd.isna(row.get("agent_2", np.nan)) else None,
            temperature=temp,
            yield_value=yield_val,
            success=(yield_val > success_threshold) if yield_val is not None else None,
            source="orderly",
            dataset="orderly_condition",
        )
        records.append(rec)

    print(f"  Processed {len(records)} valid reactions")
    return records


# ─── Buchwald HTE preprocessing ──────────────────────────────────────────────

def preprocess_buchwald_doyle(data_dir: Path, success_threshold: float = 5.0) -> list[ReactionRecord]:
    """
    Preprocess Doyle 2018 Buchwald-Hartwig amination HTE dataset.

    This dataset has a full combinatorial grid:
    - 15 aryl halides x 4 additives x 3 bases x 4 ligands = 720 combinations
    - Multiple replicates → ~3955 reactions
    - Dense yield measurements
    """
    csv_path = data_dir / "buchwald" / "doyle_2018.csv"
    if not csv_path.exists():
        print(f"Doyle 2018 data not found at {csv_path}")
        return []

    print(f"Loading Doyle 2018 Buchwald HTE from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} reactions")

    records = []
    for idx, row in df.iterrows():
        # The Doyle dataset columns vary — adapt to actual format
        yield_val = None
        for col in ["Output", "yield", "Yield", "output"]:
            if col in df.columns:
                yield_val = float(row[col]) if not pd.isna(row[col]) else None
                break

        # Build record with whatever condition columns exist
        rec = ReactionRecord(
            reaction_id=f"doyle_{idx}",
            product_smiles="",  # HTE datasets often don't have explicit SMILES
            reactant_smiles="",
            reaction_smiles="",
            ligand=str(row.get("Ligand", row.get("ligand", ""))) if "Ligand" in df.columns or "ligand" in df.columns else None,
            base=str(row.get("Base", row.get("base", ""))) if "Base" in df.columns or "base" in df.columns else None,
            solvent=str(row.get("Solvent", row.get("solvent", ""))) if "Solvent" in df.columns or "solvent" in df.columns else None,
            yield_value=yield_val,
            success=(yield_val > success_threshold) if yield_val is not None else None,
            source="buchwald_hte",
            dataset="doyle_2018",
        )
        records.append(rec)

    print(f"  Processed {len(records)} reactions")
    return records


# ─── Vocabulary building ─────────────────────────────────────────────────────

def build_condition_vocabularies(records: list[ReactionRecord], min_count: int = 3) -> dict:
    """Build vocabularies for discrete conditions from training data."""
    from collections import Counter

    vocab = {}
    for field_name in ["solvent", "catalyst", "reagent", "base", "ligand"]:
        counter = Counter()
        for rec in records:
            val = getattr(rec, field_name)
            if val and val.strip():
                counter[val.strip()] += 1

        # Filter by min count, add special tokens
        tokens = ["<PAD>", "<UNK>", "<NONE>"]
        tokens += [k for k, v in counter.most_common() if v >= min_count]
        vocab[field_name] = {tok: i for i, tok in enumerate(tokens)}

    return vocab


def build_continuous_stats(records: list[ReactionRecord]) -> dict:
    """Compute normalization statistics for continuous conditions."""
    stats = {}
    for field_name in ["temperature", "time_hours", "concentration"]:
        values = [getattr(r, field_name) for r in records if getattr(r, field_name) is not None]
        if values:
            stats[field_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
    return stats


# ─── Save/load ────────────────────────────────────────────────────────────────

def save_records(records: list[ReactionRecord], path: Path):
    """Save records as parquet for efficiency."""
    df = pd.DataFrame([asdict(r) for r in records])
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved {len(records)} records to {path}")


def load_records(path: Path) -> pd.DataFrame:
    """Load saved records."""
    return pd.read_parquet(path)


# ─── Main preprocessing pipeline ─────────────────────────────────────────────

def main():
    """Run full preprocessing pipeline."""
    data_dir = Path(__file__).parent / "raw"
    processed_dir = Path(__file__).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess each source
    all_records = []

    orderly_records = preprocess_orderly(data_dir)
    if orderly_records:
        save_records(orderly_records, processed_dir / "orderly.parquet")
        all_records.extend(orderly_records)

    doyle_records = preprocess_buchwald_doyle(data_dir)
    if doyle_records:
        save_records(doyle_records, processed_dir / "doyle_2018.parquet")
        all_records.extend(doyle_records)

    if all_records:
        # Build vocabularies from all data
        vocab = build_condition_vocabularies(all_records)
        stats = build_continuous_stats(all_records)

        with open(processed_dir / "condition_vocab.json", "w") as f:
            json.dump(vocab, f, indent=2)

        with open(processed_dir / "continuous_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nVocabulary sizes:")
        for k, v in vocab.items():
            print(f"  {k}: {len(v)}")

        print(f"\nContinuous stats:")
        for k, v in stats.items():
            print(f"  {k}: mean={v['mean']:.1f}, std={v['std']:.1f}")

    print(f"\nTotal records: {len(all_records)}")


if __name__ == "__main__":
    main()
