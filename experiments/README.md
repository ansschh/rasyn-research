# Robust Synthesis Experiments

Experiment suite for **Research Proposal 1: Robust Synthesis Planning as a Distributional, Set-Valued Decision Process**.

Tests five hypotheses at moderate scale before committing to full-scale runs.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download datasets
python data/download.py

# 3. Preprocess
python data/preprocessing.py

# 4. Pretrain shared encoder (optional but recommended)
python training/train_encoder.py

# 5. Train condition models (Experiment 2)
python training/train_condition.py --model both --split random

# 6. Train yield field ensemble (Experiment 3)
python training/train_yield_field.py --dataset doyle_2018 --n-ensemble 5

# 7. Run evaluations
python evaluation/run_eval.py --experiment all
```

## Experiments

| # | Hypothesis | Script | Key Metric |
|---|-----------|--------|------------|
| 1 | Data + baselines | `train_condition.py --model point` | Top-k exact match |
| 2 | Set-valued conditions beat point conditions | `train_condition.py --model ebm` | Coverage@0.9 |
| 3 | Yield field + uncertainty improves selection | `train_yield_field.py` | Robust utility |
| 4 | Risk-sensitive route scoring survives perturbation | `evaluation/run_eval.py --experiment route` | Success rate under perturbation |
| 5 | Boundary-focused AL is more data-efficient | `evaluation/run_eval.py --experiment active` | Experiments to target IoU |

## Architecture

```
ReactionEncoder (shared, d=256, 4 layers, ~2M params)
    |
    +-> PointConditionPredictor (baseline)
    |       reaction -> (solvent, catalyst, reagent, base, temp)
    |
    +-> ConditionManifoldEBM (novel)
    |       (reaction, condition) -> energy E(T,c)
    |       Low-energy region = viable conditions
    |
    +-> YieldFieldEnsemble (novel)
            (reaction, condition) -> (yield, feasibility, uncertainty)
            Deep ensemble (5 members) for epistemic uncertainty
```

## Datasets

- **ORDerly**: ~1M cleaned reactions from Open Reaction Database (conditions + yields)
- **Buchwald HTE (Doyle 2018)**: ~3955 dense condition-outcome measurements (ideal for field modeling)

## Key Files

```
config/           - YAML configurations (model sizes, training params)
data/             - Download, preprocessing, splits, featurization
models/           - All model architectures
training/         - Training scripts
evaluation/       - Metrics and evaluation runners
notebooks/        - Analysis and visualization
```
