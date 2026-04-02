"""
Preprocessing for spectral data.

Converts molecules + spectra into standardized format:
- Molecular graphs (node/edge features from RDKit)
- Peak sets (sparse measure representation)
- Paired (molecule, spectrum) records
"""

import json
import ast
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class MoleculeRecord:
    """Molecule with computed features."""
    mol_id: str
    smiles: str
    num_atoms: int
    num_heavy: int
    mol_weight: float
    # Atom features as JSON-serialized lists
    atom_types: str     # list of atomic numbers
    atom_aromatic: str  # list of booleans
    atom_formal_charge: str
    atom_hybridization: str
    atom_num_hs: str

@dataclass
class NMRRecord:
    """NMR spectrum record."""
    mol_id: str
    smiles: str
    nucleus: str           # "1H" or "13C"
    shifts: str            # JSON list of shifts (ppm)
    n_peaks: int
    source: str = ""

@dataclass
class MSMSRecord:
    """MS/MS spectrum record."""
    mol_id: str
    smiles: str
    precursor_mz: float
    adduct: str
    collision_energy: float
    mz_values: str         # JSON list
    intensities: str       # JSON list
    n_peaks: int
    source: str = ""


# ─── Molecule featurization ──────────────────────────────────────────────────

ATOM_TYPES = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]  # H, B, C, N, O, F, Si, P, S, Cl, Br, I

def featurize_molecule(smiles: str) -> Optional[dict]:
    """Extract atom-level features from a SMILES string."""
    if not HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)

    atom_types = []
    atom_aromatic = []
    atom_charge = []
    atom_hyb = []
    atom_hs = []

    for atom in mol.GetAtoms():
        atom_types.append(atom.GetAtomicNum())
        atom_aromatic.append(int(atom.GetIsAromatic()))
        atom_charge.append(atom.GetFormalCharge())
        atom_hyb.append(str(atom.GetHybridization()))
        atom_hs.append(atom.GetTotalNumHs())

    return {
        "num_atoms": mol.GetNumAtoms(),
        "num_heavy": mol.GetNumHeavyAtoms(),
        "mol_weight": Descriptors.MolWt(mol),
        "atom_types": json.dumps(atom_types),
        "atom_aromatic": json.dumps(atom_aromatic),
        "atom_formal_charge": json.dumps(atom_charge),
        "atom_hybridization": json.dumps(atom_hyb),
        "atom_num_hs": json.dumps(atom_hs),
    }


def mol_to_graph(smiles: str) -> Optional[dict]:
    """
    Convert SMILES to graph tensors (node features, edge index, edge features).
    Returns numpy arrays ready for PyTorch conversion.
    """
    if not HAS_RDKIT:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Node features: [atomic_num_onehot, aromatic, charge, degree, num_hs, hybridization_onehot]
    n_atoms = mol.GetNumAtoms()
    node_feats = np.zeros((n_atoms, 13), dtype=np.float32)

    for i, atom in enumerate(mol.GetAtoms()):
        # Atomic number one-hot (common elements)
        anum = atom.GetAtomicNum()
        if anum in ATOM_TYPES:
            node_feats[i, ATOM_TYPES.index(anum)] = 1.0
        # Aromatic
        node_feats[i, len(ATOM_TYPES)] = float(atom.GetIsAromatic())

    # Edge index and features
    edges_src, edges_dst = [], []
    edge_feats = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges_src.extend([i, j])
        edges_dst.extend([j, i])
        bt = bond.GetBondTypeAsDouble()
        feat = [bt, float(bond.GetIsAromatic()), float(bond.IsInRing())]
        edge_feats.extend([feat, feat])  # both directions

    edge_index = np.array([edges_src, edges_dst], dtype=np.int64) if edges_src else np.zeros((2, 0), dtype=np.int64)
    edge_features = np.array(edge_feats, dtype=np.float32) if edge_feats else np.zeros((0, 3), dtype=np.float32)

    return {
        "node_features": node_feats,
        "edge_index": edge_index,
        "edge_features": edge_features,
        "n_atoms": n_atoms,
    }


# ─── Peak set representation ─────────────────────────────────────────────────

def shifts_to_peak_set(shifts: list[float], nucleus: str = "13C") -> np.ndarray:
    """
    Convert a list of chemical shifts to a peak set representation.
    Each peak: [position, intensity, nucleus_id, uncertainty]
    """
    n = len(shifts)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)

    peaks = np.zeros((n, 4), dtype=np.float32)
    peaks[:, 0] = shifts                            # position (ppm)
    peaks[:, 1] = 1.0                               # intensity (uniform for 1D NMR)
    peaks[:, 2] = 0.0 if nucleus == "1H" else 1.0   # nucleus type
    peaks[:, 3] = 0.5                               # default uncertainty (ppm)

    return peaks


def msms_to_peak_set(mz_values: list[float], intensities: list[float]) -> np.ndarray:
    """
    Convert MS/MS spectrum to peak set representation.
    Each peak: [mz, intensity, modality_id=2, uncertainty]
    """
    n = len(mz_values)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)

    # Normalize intensities
    max_int = max(intensities) if intensities else 1.0
    norm_int = [i / max_int for i in intensities]

    peaks = np.zeros((n, 4), dtype=np.float32)
    peaks[:, 0] = mz_values
    peaks[:, 1] = norm_int
    peaks[:, 2] = 2.0           # modality: MS/MS
    peaks[:, 3] = 0.01          # m/z uncertainty (Da)

    return peaks


# ─── Preprocessing pipelines ─────────────────────────────────────────────────

def preprocess_synthetic_nmr(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess synthetic NMR data."""
    path = data_dir / "synthetic_nmr" / "synthetic_nmr.parquet"
    if not path.exists():
        print(f"Synthetic NMR not found at {path}. Run download.py first.")
        return pd.DataFrame(), pd.DataFrame()

    print(f"Loading synthetic NMR from {path}...")
    df = pd.read_parquet(path)
    print(f"  Loaded {len(df)} molecules")

    # Build molecule records
    mol_records = []
    nmr_records = []

    for idx, row in df.iterrows():
        smiles = row["smiles"]
        feats = featurize_molecule(smiles)
        if feats is None:
            continue

        mol_id = f"synth_{idx}"
        mol_records.append(MoleculeRecord(
            mol_id=mol_id,
            smiles=smiles,
            **feats,
        ))

        # 13C shifts
        c_shifts = ast.literal_eval(row["c13_shifts"]) if isinstance(row["c13_shifts"], str) else row["c13_shifts"]
        if c_shifts:
            nmr_records.append(NMRRecord(
                mol_id=mol_id,
                smiles=smiles,
                nucleus="13C",
                shifts=json.dumps(c_shifts),
                n_peaks=len(c_shifts),
                source="synthetic",
            ))

        # 1H shifts
        h_shifts = ast.literal_eval(row["h1_shifts"]) if isinstance(row["h1_shifts"], str) else row["h1_shifts"]
        if h_shifts:
            nmr_records.append(NMRRecord(
                mol_id=mol_id,
                smiles=smiles,
                nucleus="1H",
                shifts=json.dumps(h_shifts),
                n_peaks=len(h_shifts),
                source="synthetic",
            ))

    mol_df = pd.DataFrame([asdict(r) for r in mol_records])
    nmr_df = pd.DataFrame([asdict(r) for r in nmr_records])

    print(f"  {len(mol_records)} molecules, {len(nmr_records)} NMR records")
    return mol_df, nmr_df


def main():
    data_dir = Path(__file__).parent / "raw"
    processed_dir = Path(__file__).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Process synthetic NMR
    mol_df, nmr_df = preprocess_synthetic_nmr(data_dir)

    if len(mol_df) > 0:
        mol_df.to_parquet(processed_dir / "molecules.parquet", index=False)
        print(f"Saved {len(mol_df)} molecules to molecules.parquet")

    if len(nmr_df) > 0:
        nmr_df.to_parquet(processed_dir / "nmr_spectra.parquet", index=False)
        print(f"Saved {len(nmr_df)} NMR records to nmr_spectra.parquet")

    print(f"\nPreprocessing complete.")


if __name__ == "__main__":
    main()
