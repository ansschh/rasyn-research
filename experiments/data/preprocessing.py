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
    Preprocess ORDerly condition dataset (Figshare benchmark parquets).

    Actual columns (from orderly_condition_train.parquet):
    - rxn_str: full reaction SMILES (reactants>>products)
    - solvent_000, solvent_001: solvents
    - agent_000, agent_001, agent_002: reagents/agents
    - product_000: product SMILES
    - reactant_000, reactant_001: reactant SMILES
    - temperature: temperature in Celsius
    - rxn_time: reaction time
    - yield_000: yield value
    """
    # Try parquet first (Figshare benchmark), then csv.gz fallback
    parquet_path = data_dir / "orderly" / "orderly_condition_train.parquet"
    if parquet_path.exists():
        print(f"Loading ORDerly from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
    else:
        csv_path = data_dir / "orderly" / "orderly_condition_data.csv.gz"
        if not csv_path.exists():
            print(f"ORDerly data not found. Run download.py first.")
            return []
        print(f"Loading ORDerly from {csv_path}...")
        df = pd.read_csv(csv_path, compression="gzip")

    print(f"  Loaded {len(df)} reactions")

    records = []
    for idx, row in df.iterrows():
        # Parse reaction SMILES
        rxn_smi = row.get("rxn_str", "")
        if not rxn_smi or pd.isna(rxn_smi):
            continue

        parts = str(rxn_smi).split(">>")
        if len(parts) != 2:
            continue

        reactants_smi, products_smi = parts[0], parts[1]
        product_can = canonicalize_smiles(products_smi)
        reactant_can = canonicalize_smiles(reactants_smi)

        if product_can is None or reactant_can is None:
            continue

        # Yield
        yield_val = row.get("yield_000", None)
        if yield_val is not None and not pd.isna(yield_val):
            yield_val = float(yield_val)
        else:
            yield_val = None

        # Temperature
        temp = row.get("temperature", None)
        if temp is not None and not pd.isna(temp):
            temp = float(temp)
        else:
            temp = None

        # Reaction time
        rxn_time = row.get("rxn_time", None)
        if rxn_time is not None and not pd.isna(rxn_time):
            rxn_time = float(rxn_time)
        else:
            rxn_time = None

        # Extract condition strings safely
        def safe_str(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            s = str(val).strip()
            return s if s else None

        rec = ReactionRecord(
            reaction_id=f"orderly_{idx}",
            product_smiles=product_can,
            reactant_smiles=reactant_can,
            reaction_smiles=f"{product_can}>>{reactant_can}",
            solvent=safe_str(row.get("solvent_000")),
            catalyst=None,  # ORDerly benchmark doesn't separate catalyst
            reagent=safe_str(row.get("agent_000")),
            base=safe_str(row.get("agent_001")),
            ligand=safe_str(row.get("agent_002")),
            temperature=temp,
            time_hours=rxn_time,
            yield_value=yield_val,
            success=(yield_val > success_threshold) if yield_val is not None else None,
            source="orderly",
            dataset="orderly_condition",
        )
        records.append(rec)

        if idx % 100000 == 0 and idx > 0:
            print(f"  Processed {idx} reactions...")

    print(f"  Processed {len(records)} valid reactions")
    return records


# ─── Buchwald HTE preprocessing ──────────────────────────────────────────────

def preprocess_buchwald_doyle(data_dir: Path, success_threshold: float = 5.0) -> list[ReactionRecord]:
    """
    Preprocess Doyle 2018 Buchwald-Hartwig amination HTE dataset.

    Actual columns (from doyle_2018.csv / data_table.csv):
    - base, base_smiles: base name and SMILES
    - ligand, ligand_smiles: ligand name and SMILES
    - aryl_halide, aryl_halide_smiles: substrate
    - additive, additive_smiles: additive
    - product_smiles: product SMILES
    - yield: reaction yield (0-100)
    - plate, row, col: plate position

    4599 reactions in a combinatorial grid — ideal for field modeling.
    """
    csv_path = data_dir / "buchwald" / "doyle_2018.csv"
    if not csv_path.exists():
        print(f"Doyle 2018 data not found at {csv_path}")
        return []

    print(f"Loading Doyle 2018 Buchwald HTE from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} reactions")

    def safe_str(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        s = str(val).strip()
        return s if s else None

    records = []
    for idx, row in df.iterrows():
        yield_val = row.get("yield", None)
        if yield_val is not None and not pd.isna(yield_val):
            yield_val = float(yield_val)
        else:
            yield_val = None

        # Product SMILES
        product = safe_str(row.get("product_smiles"))
        product_can = canonicalize_smiles(product) if product else ""

        # Build reaction SMILES from aryl_halide + amine (implicit) -> product
        aryl_halide = safe_str(row.get("aryl_halide_smiles"))

        rec = ReactionRecord(
            reaction_id=f"doyle_{idx}",
            product_smiles=product_can or "",
            reactant_smiles=aryl_halide or "",
            reaction_smiles=f"{product_can}>>{aryl_halide}" if product_can and aryl_halide else "",
            ligand=safe_str(row.get("ligand")),
            base=safe_str(row.get("base")),
            reagent=safe_str(row.get("additive")),
            catalyst=None,  # Pd catalyst is implicit (same across all)
            solvent=None,    # Same solvent across all in this dataset
            yield_value=yield_val,
            success=(yield_val > success_threshold) if yield_val is not None else None,
            source="buchwald_hte",
            dataset="doyle_2018",
        )
        records.append(rec)

    n_success = sum(1 for r in records if r.success)
    print(f"  Processed {len(records)} reactions ({n_success} successful, {len(records)-n_success} failed)")
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
