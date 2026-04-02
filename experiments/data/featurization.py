"""
Featurization: convert reaction records into model-ready tensors.

Handles:
- SMILES tokenization (regex-based, atom-level)
- Condition encoding (discrete + continuous)
- Fingerprint computation
- Dataset / DataLoader creation
"""

import re
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ─── SMILES tokenizer ────────────────────────────────────────────────────────

# Regex pattern for atom-level SMILES tokenization
SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]"           # bracketed atoms [nH], [C@@H]
    r"|Br|Cl"                # two-letter elements
    r"|[A-Z][a-z]?"          # one/two-letter uppercase elements
    r"|[a-z]"                # aromatic atoms (c, n, o, s, p, b)
    r"|[0-9]"                # ring closure digits
    r"|[%()\-=#+/\\@.:~]"   # bonds, branches, stereochemistry
    r")"
)

SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>", "<SEP>"]


class SmilesTokenizer:
    """Regex-based SMILES tokenizer producing atom-level tokens."""

    def __init__(self, vocab: Optional[dict[str, int]] = None):
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def tokenize(self, smiles: str) -> list[str]:
        """Tokenize a SMILES string into atom-level tokens."""
        return SMILES_REGEX.findall(smiles)

    def encode(self, smiles: str, max_len: int = 256) -> list[int]:
        """Encode SMILES to token ids."""
        tokens = self.tokenize(smiles)
        ids = [self.vocab.get("<BOS>", 2)]
        for t in tokens[:max_len - 2]:
            ids.append(self.vocab.get(t, self.vocab.get("<UNK>", 1)))
        ids.append(self.vocab.get("<EOS>", 3))
        return ids

    def build_vocab(self, smiles_list: list[str], min_count: int = 1) -> dict[str, int]:
        """Build vocabulary from a list of SMILES strings."""
        from collections import Counter
        counter = Counter()
        for smi in smiles_list:
            if smi and not pd.isna(smi):
                counter.update(self.tokenize(str(smi)))

        self.vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        idx = len(SPECIAL_TOKENS)
        for tok, count in counter.most_common():
            if count >= min_count:
                self.vocab[tok] = idx
                idx += 1

        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        return self.vocab

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "SmilesTokenizer":
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab)


# ─── Condition encoder ────────────────────────────────────────────────────────

class ConditionEncoder:
    """Encode discrete + continuous conditions into tensors."""

    def __init__(
        self,
        discrete_vocabs: dict[str, dict[str, int]],
        continuous_stats: dict[str, dict[str, float]],
    ):
        self.discrete_vocabs = discrete_vocabs
        self.continuous_stats = continuous_stats
        self.discrete_fields = list(discrete_vocabs.keys())
        self.continuous_fields = list(continuous_stats.keys())

    def encode_discrete(self, record: dict) -> torch.LongTensor:
        """Encode discrete conditions as vocab indices."""
        ids = []
        for field in self.discrete_fields:
            val = record.get(field)
            vocab = self.discrete_vocabs[field]
            if val and val in vocab:
                ids.append(vocab[val])
            elif val:
                ids.append(vocab.get("<UNK>", 1))
            else:
                ids.append(vocab.get("<NONE>", 2))
        return torch.tensor(ids, dtype=torch.long)

    def encode_continuous(self, record: dict) -> torch.FloatTensor:
        """Encode continuous conditions, normalized to ~N(0,1)."""
        vals = []
        masks = []
        for field in self.continuous_fields:
            val = record.get(field)
            stats = self.continuous_stats[field]
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                normalized = (float(val) - stats["mean"]) / max(stats["std"], 1e-6)
                vals.append(normalized)
                masks.append(1.0)
            else:
                vals.append(0.0)
                masks.append(0.0)
        return torch.tensor(vals, dtype=torch.float32), torch.tensor(masks, dtype=torch.float32)

    @property
    def n_discrete(self) -> int:
        return len(self.discrete_fields)

    @property
    def n_continuous(self) -> int:
        return len(self.continuous_fields)


# ─── Fingerprints ─────────────────────────────────────────────────────────────

def compute_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048) -> Optional[np.ndarray]:
    """Compute Morgan fingerprint as numpy array."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.float32)
    except Exception:
        return None


# ─── PyTorch Datasets ─────────────────────────────────────────────────────────

class ReactionConditionDataset(Dataset):
    """Dataset for condition prediction experiments."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: SmilesTokenizer,
        condition_encoder: ConditionEncoder,
        max_seq_len: int = 256,
        use_fingerprints: bool = False,
        fp_bits: int = 2048,
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.condition_encoder = condition_encoder
        self.max_seq_len = max_seq_len
        self.use_fingerprints = use_fingerprints
        self.fp_bits = fp_bits

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx].to_dict()

        # Encode reaction SMILES
        rxn_smi = row.get("reaction_smiles", "") or row.get("product_smiles", "")
        token_ids = self.tokenizer.encode(str(rxn_smi), max_len=self.max_seq_len)

        # Pad to max_seq_len
        pad_id = self.tokenizer.vocab.get("<PAD>", 0)
        attention_mask = [1] * len(token_ids) + [0] * (self.max_seq_len - len(token_ids))
        token_ids = token_ids + [pad_id] * (self.max_seq_len - len(token_ids))

        item = {
            "input_ids": torch.tensor(token_ids[:self.max_seq_len], dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask[:self.max_seq_len], dtype=torch.float32),
        }

        # Encode conditions (targets for condition prediction, inputs for yield prediction)
        item["discrete_conditions"] = self.condition_encoder.encode_discrete(row)
        cont_vals, cont_mask = self.condition_encoder.encode_continuous(row)
        item["continuous_conditions"] = cont_vals
        item["continuous_mask"] = cont_mask

        # Yield / feasibility targets
        yield_val = row.get("yield_value")
        if yield_val is not None and not (isinstance(yield_val, float) and np.isnan(yield_val)):
            item["yield"] = torch.tensor(float(yield_val) / 100.0, dtype=torch.float32)  # normalize to [0,1]
            item["has_yield"] = torch.tensor(1.0, dtype=torch.float32)
        else:
            item["yield"] = torch.tensor(0.0, dtype=torch.float32)
            item["has_yield"] = torch.tensor(0.0, dtype=torch.float32)

        success = row.get("success")
        if success is not None and not (isinstance(success, float) and np.isnan(success)):
            item["success"] = torch.tensor(float(success), dtype=torch.float32)
        else:
            item["success"] = torch.tensor(0.0, dtype=torch.float32)

        # Optional: fingerprints
        if self.use_fingerprints:
            fp = compute_morgan_fp(str(rxn_smi).split(">>")[0], n_bits=self.fp_bits)
            if fp is not None:
                item["fingerprint"] = torch.tensor(fp, dtype=torch.float32)
            else:
                item["fingerprint"] = torch.zeros(self.fp_bits, dtype=torch.float32)

        return item


def collate_fn(batch: list[dict]) -> dict:
    """Collate a batch of items."""
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


def create_dataloader(
    dataset: ReactionConditionDataset,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader with proper collation."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )
