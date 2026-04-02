# Spectral Inference Experiments

Experiment suite for **Research Proposal 2: Amortized Bayesian Spectral Inference Across NMR and MS/MS**.

Tests whether treating structure elucidation as a posterior inference problem over peak sets outperforms deterministic modality-specific baselines.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets + generate synthetic NMR data
python data/download.py

# 3. Preprocess
python data/preprocessing.py

# 4. Train forward NMR model (Experiment 1)
python training/train_forward_nmr.py --nucleus 13C

# 5. Run evaluations
python evaluation/spectral_metrics.py
```

## Experiments

| # | Hypothesis | Script | Key Metric |
|---|-----------|--------|------------|
| 1 | Forward NMR with calibrated uncertainty | `train_forward_nmr.py` | Shift MAE + calibration |
| 2 | Forward MS/MS predictor | (pending real MS data) | Cosine similarity |
| 3 | Posterior > point prediction | `posterior.py` | Top-k recovery + calibration |
| 4 | NMR+MS/MS > either alone | `posterior.py` (ablation) | Entropy reduction |
| 5 | Active measurement policy | `measurement_policy.py` | Measurements to identify |

## Architecture

```
MolecularGraphEncoder (MPNN, 4 layers, ~2M params)
    |
    +-> ForwardNMRModel (structure → per-atom shifts + uncertainty)
    |
    +-> ForwardMSMSModel (structure + CE → fragmentation spectrum)

PeakSetEncoder (Set Transformer with ISAB, ~1.5M params)
    |
    +-> Spectral embedding (permutation-invariant, modality-aware)

PosteriorInferenceEngine
    |
    +-> Scores candidates via forward model consistency + OT matching
    +-> Returns calibrated posterior over structure hypotheses

MeasurementPolicy
    +-> Selects next most informative spectrum to acquire
```

## Key Innovation: Peak Sets, Not Sequences

Spectra are NOT tokenized into sequences. They are treated as **unordered sets of peaks** with features `[position, intensity, modality, uncertainty]`. This uses Set Transformer (ISAB) for O(n*m) permutation-invariant encoding — more physically faithful than positional-encoded sequence models.
