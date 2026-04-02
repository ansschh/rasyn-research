"""
Preprocessing for real spectral datasets.

Sources:
1. nmrshiftdb2 (HuggingFace: structure-epflai/nmrshiftdb2) — 57K molecules, experimental NMR
2. MassSpecGym (HuggingFace: roman-bushuiev/MassSpecGym) — 231K MS/MS spectra

Produces standardized formats:
- Molecular graphs (node/edge features from RDKit)
- Peak sets (sparse measure representation)
- Paired (molecule, spectrum) records
"""

import json
import re
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

# ─── Molecular graph featurization ───────────────────────────────────────────

ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]


def mol_to_graph(smiles: str) -> Optional[dict]:
    """Convert SMILES to graph tensors."""
    if not HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    n_atoms = mol.GetNumAtoms()
    node_feats = np.zeros((n_atoms, len(ATOM_TYPES) + 1), dtype=np.float32)

    for i, atom in enumerate(mol.GetAtoms()):
        anum = atom.GetAtomicNum()
        if anum in ATOM_TYPES:
            node_feats[i, ATOM_TYPES.index(anum)] = 1.0
        node_feats[i, len(ATOM_TYPES)] = float(atom.GetIsAromatic())

    edges_src, edges_dst, edge_feats = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges_src.extend([i, j])
        edges_dst.extend([j, i])
        feat = [bond.GetBondTypeAsDouble(), float(bond.GetIsAromatic()), float(bond.IsInRing())]
        edge_feats.extend([feat, feat])

    edge_index = np.array([edges_src, edges_dst], dtype=np.int64) if edges_src else np.zeros((2, 0), dtype=np.int64)
    edge_features = np.array(edge_feats, dtype=np.float32) if edge_feats else np.zeros((0, 3), dtype=np.float32)

    return {"node_features": node_feats, "edge_index": edge_index, "edge_features": edge_features, "n_atoms": n_atoms}


def load_records(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


# ─── nmrshiftdb2 preprocessing ───────────────────────────────────────────────

def parse_nmrshiftdb2_spectrum(spectrum_str: str) -> Optional[list[tuple[float, float]]]:
    """
    Parse nmrshiftdb2 spectrum string format.
    Format varies but typically: "shift1|intensity1|atomidx1|shift2|intensity2|atomidx2|..."
    or pipe-separated shift values.
    """
    if not spectrum_str or pd.isna(spectrum_str):
        return None

    spectrum_str = str(spectrum_str).strip()
    if not spectrum_str:
        return None

    peaks = []
    try:
        # Split by pipe
        parts = spectrum_str.split("|")
        # Try groups of 3 (shift, intensity, atom_index)
        if len(parts) >= 3:
            i = 0
            while i + 2 < len(parts):
                try:
                    shift = float(parts[i].strip())
                    intensity = float(parts[i + 1].strip())
                    peaks.append((shift, intensity))
                    i += 3
                except (ValueError, IndexError):
                    i += 1
            # Fallback: if we got nothing, try groups of 2
            if not peaks:
                i = 0
                while i + 1 < len(parts):
                    try:
                        shift = float(parts[i].strip())
                        intensity = float(parts[i + 1].strip())
                        peaks.append((shift, intensity))
                        i += 2
                    except (ValueError, IndexError):
                        i += 1
            # Last fallback: every float is a shift
            if not peaks:
                for p in parts:
                    try:
                        shift = float(p.strip())
                        peaks.append((shift, 1.0))
                    except ValueError:
                        continue
    except Exception:
        return None

    return peaks if peaks else None


def preprocess_nmrshiftdb2(data_dir: Path) -> pd.DataFrame:
    """
    Preprocess nmrshiftdb2 into (smiles, nucleus, shifts, n_peaks) records.

    The dataset has columns like "Spectrum 13C 0", "Spectrum 1H 0", etc.
    Each spectrum column contains pipe-separated shift data.
    """
    parquet_path = data_dir / "nmrshiftdb2" / "data" / "nmrshiftdb2.parquet"
    if not parquet_path.exists():
        print(f"nmrshiftdb2 not found at {parquet_path}")
        return pd.DataFrame()

    print(f"Loading nmrshiftdb2 from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df)} molecules with {len(df.columns)} columns")

    # Find spectrum columns
    spectrum_cols = {}
    for col in df.columns:
        if col.startswith("Spectrum ") and "rawdata" not in col.lower():
            # Extract nucleus: "Spectrum 13C 0" -> "13C"
            parts = col.replace("Spectrum ", "").split()
            if len(parts) >= 1:
                nucleus = parts[0]
                if nucleus not in spectrum_cols:
                    spectrum_cols[nucleus] = []
                spectrum_cols[nucleus].append(col)

    print(f"  Found nuclei: {list(spectrum_cols.keys())}")
    for nuc, cols in spectrum_cols.items():
        print(f"    {nuc}: {len(cols)} spectrum columns")

    # Extract NMR records
    records = []
    for idx, row in df.iterrows():
        smiles = row.get("SMILES")
        if not smiles or pd.isna(smiles):
            continue

        # Canonicalize
        if HAS_RDKIT:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue
            can_smiles = Chem.MolToSmiles(mol, canonical=True)
        else:
            can_smiles = str(smiles)

        solvent = row.get("Solvent", "")
        temp_k = row.get("Temperature [K]", None)
        field_mhz = row.get("Field Strength [MHz]", None)

        # Process each nucleus
        for nucleus in ["13C", "1H"]:
            if nucleus not in spectrum_cols:
                continue

            for col in spectrum_cols[nucleus]:
                spec_str = row.get(col)
                peaks = parse_nmrshiftdb2_spectrum(spec_str)
                if peaks and len(peaks) > 0:
                    shifts = [p[0] for p in peaks]
                    intensities = [p[1] for p in peaks]

                    records.append({
                        "mol_id": f"nmrdb_{idx}",
                        "smiles": can_smiles,
                        "nucleus": nucleus,
                        "shifts": json.dumps(shifts),
                        "intensities": json.dumps(intensities),
                        "n_peaks": len(peaks),
                        "solvent": str(solvent) if not pd.isna(solvent) else "",
                        "temperature_k": float(temp_k) if temp_k and not pd.isna(temp_k) else None,
                        "field_mhz": float(field_mhz) if field_mhz and not pd.isna(field_mhz) else None,
                        "source": "nmrshiftdb2",
                        "spectrum_col": col,
                    })
                    break  # Take first valid spectrum per nucleus per molecule

        if idx % 10000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(df)} molecules...")

    result_df = pd.DataFrame(records)
    print(f"  Extracted {len(result_df)} NMR spectrum records")

    # Summary
    for nuc in ["13C", "1H"]:
        nuc_df = result_df[result_df["nucleus"] == nuc]
        if len(nuc_df) > 0:
            avg_peaks = nuc_df["n_peaks"].mean()
            print(f"    {nuc}: {len(nuc_df)} spectra, avg {avg_peaks:.1f} peaks")

    return result_df


# ─── MassSpecGym preprocessing ───────────────────────────────────────────────

def preprocess_massspecgym(data_dir: Path) -> pd.DataFrame:
    """
    Preprocess MassSpecGym into (smiles, mz_values, intensities, ...) records.

    Columns: identifier, mzs, intensities, smiles, inchikey, formula,
             precursor_mz, adduct, collision_energy, fold
    """
    tsv_path = data_dir / "massspecgym" / "data" / "MassSpecGym.tsv"
    if not tsv_path.exists():
        print(f"MassSpecGym not found at {tsv_path}")
        return pd.DataFrame()

    print(f"Loading MassSpecGym from {tsv_path}...")
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"  Loaded {len(df)} MS/MS spectra")

    records = []
    for idx, row in df.iterrows():
        smiles = row.get("smiles")
        if not smiles or pd.isna(smiles):
            continue

        # Canonicalize
        if HAS_RDKIT:
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                continue
            can_smiles = Chem.MolToSmiles(mol, canonical=True)
        else:
            can_smiles = str(smiles)

        # Parse mz and intensities (stored as space or comma-separated strings)
        mzs_str = str(row.get("mzs", ""))
        ints_str = str(row.get("intensities", ""))

        try:
            mzs = [float(x) for x in mzs_str.replace(",", " ").split() if x.strip()]
            ints = [float(x) for x in ints_str.replace(",", " ").split() if x.strip()]
        except ValueError:
            continue

        if len(mzs) == 0 or len(mzs) != len(ints):
            continue

        precursor_mz = row.get("precursor_mz", 0)
        if pd.isna(precursor_mz):
            precursor_mz = 0

        ce = row.get("collision_energy", 0)
        if pd.isna(ce):
            ce = 0

        records.append({
            "mol_id": f"msg_{idx}",
            "smiles": can_smiles,
            "inchikey": str(row.get("inchikey", "")),
            "formula": str(row.get("formula", "")),
            "precursor_mz": float(precursor_mz),
            "adduct": str(row.get("adduct", "")),
            "collision_energy": float(ce) if not isinstance(ce, str) else 0,
            "mz_values": json.dumps(mzs),
            "intensities": json.dumps(ints),
            "n_peaks": len(mzs),
            "fold": str(row.get("fold", "")),
            "source": "massspecgym",
        })

        if idx % 50000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(df)} spectra...")

    result_df = pd.DataFrame(records)
    print(f"  Extracted {len(result_df)} valid MS/MS records")
    print(f"    Avg peaks: {result_df['n_peaks'].mean():.1f}")
    print(f"    Unique molecules: {result_df['smiles'].nunique()}")

    return result_df


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    data_dir = Path(__file__).parent / "raw"
    processed_dir = Path(__file__).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Process NMR
    nmr_df = preprocess_nmrshiftdb2(data_dir)
    if len(nmr_df) > 0:
        nmr_df.to_parquet(processed_dir / "nmr_spectra.parquet", index=False)
        print(f"Saved {len(nmr_df)} NMR records")

    # Process MS/MS
    msms_df = preprocess_massspecgym(data_dir)
    if len(msms_df) > 0:
        msms_df.to_parquet(processed_dir / "msms_spectra.parquet", index=False)
        print(f"Saved {len(msms_df)} MS/MS records")

    # Find molecules with BOTH NMR and MS/MS
    if len(nmr_df) > 0 and len(msms_df) > 0:
        nmr_smiles = set(nmr_df["smiles"].unique())
        msms_smiles = set(msms_df["smiles"].unique())
        overlap = nmr_smiles & msms_smiles
        print(f"\nMultimodal overlap: {len(overlap)} molecules have BOTH NMR and MS/MS")
        if overlap:
            overlap_df = pd.DataFrame({"smiles": list(overlap)})
            overlap_df.to_parquet(processed_dir / "multimodal_overlap.parquet", index=False)

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
