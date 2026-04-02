"""
Train Yield-Feasibility Field Model (Experiment 3).

Tests H2: Does modeling yield as a field over condition space improve route selection?

Uses Buchwald HTE datasets (dense condition grids) for faithful evaluation.
Trains a deep ensemble for epistemic uncertainty estimation.
"""

import sys
import json
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.featurization import SmilesTokenizer, ConditionEncoder, ReactionConditionDataset, create_dataloader
from data.preprocessing import load_records
from data.splits import get_splits
from models.reaction_encoder import ReactionEncoder
from models.condition_manifold import ConditionEmbedder
from models.yield_field import YieldFeasibilityField, YieldFieldEnsemble
from training.utils import set_seed, get_device, Trainer, load_checkpoint, save_checkpoint


def yield_field_loss_fn(model_parts, batch):
    """
    Loss for yield field model.
    model_parts = (encoder, condition_embedder, yield_model)
    """
    encoder, cond_embedder, yield_model = model_parts

    # Encode reaction
    enc_out = encoder(batch["input_ids"], batch["attention_mask"])
    reaction_repr = enc_out["pooled"]

    # Encode conditions
    condition_repr = cond_embedder(batch["discrete_conditions"], batch["continuous_conditions"])

    # Predict yield field
    preds = yield_model(reaction_repr, condition_repr)

    # Compute loss
    losses = yield_model.compute_loss(
        preds,
        yield_target=batch["yield"],
        has_yield=batch["has_yield"],
        success_target=batch["success"],
        reaction_repr=reaction_repr,
        condition_repr=condition_repr,
    )
    return losses


class YieldFieldBundle(torch.nn.Module):
    """Wraps encoder + condition_embedder + yield_model."""
    def __init__(self, encoder, cond_embedder, yield_model):
        super().__init__()
        self.encoder = encoder
        self.cond_embedder = cond_embedder
        self.yield_model = yield_model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["orderly", "doyle_2018"], default="doyle_2018",
                        help="Dataset to train on (doyle_2018 is dense HTE, ideal for field modeling)")
    parser.add_argument("--split", choices=["random", "scaffold"], default="random")
    parser.add_argument("--n-ensemble", type=int, default=5, help="Number of ensemble members")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load configs
    base_cfg_path = Path(__file__).parent.parent / "config" / "base.yaml"
    yield_cfg_path = Path(__file__).parent.parent / "config" / "yield_field.yaml"
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)
    with open(yield_cfg_path) as f:
        yield_cfg = yaml.safe_load(f)

    device = get_device()
    print(f"Device: {device}")

    # Load data
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    df = load_records(processed_dir / f"{args.dataset}.parquet")
    print(f"Loaded {len(df)} reactions from {args.dataset}")

    # Filter to reactions with yield data
    if "yield_value" in df.columns:
        df = df[df["yield_value"].notna()].reset_index(drop=True)
        print(f"  {len(df)} reactions with yield data")

    # Load tokenizer + condition encoder
    tokenizer = SmilesTokenizer.load(processed_dir / "smiles_vocab.json") if (processed_dir / "smiles_vocab.json").exists() else SmilesTokenizer()
    if not tokenizer.vocab:
        tokenizer.build_vocab(df["reaction_smiles"].dropna().tolist() if "reaction_smiles" in df.columns else [])

    with open(processed_dir / "condition_vocab.json") as f:
        discrete_vocabs = json.load(f)
    with open(processed_dir / "continuous_stats.json") as f:
        continuous_stats = json.load(f)
    cond_encoder = ConditionEncoder(discrete_vocabs, continuous_stats)

    discrete_vocab_sizes = {k: len(v) for k, v in discrete_vocabs.items()}
    enc_cfg = base_cfg["encoder"]
    field_cfg = yield_cfg["field"]
    train_cfg = yield_cfg.get("training", base_cfg["training"])

    # ─── Train ensemble ──────────────────────────────────────────────────

    for member_idx in range(args.n_ensemble):
        member_seed = args.seed + member_idx * 1000
        set_seed(member_seed)
        print(f"\n{'='*60}")
        print(f"Training ensemble member {member_idx + 1}/{args.n_ensemble} (seed={member_seed})")
        print(f"{'='*60}")

        # Fresh split for each seed (same strategy, different init)
        splits = get_splits(df, strategy=args.split, seed=args.seed)  # same split, different model init

        train_ds = ReactionConditionDataset(splits["train"], tokenizer, cond_encoder, max_seq_len=enc_cfg["max_seq_len"])
        val_ds = ReactionConditionDataset(splits["val"], tokenizer, cond_encoder, max_seq_len=enc_cfg["max_seq_len"])

        train_loader = create_dataloader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
        val_loader = create_dataloader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)

        # Create models
        encoder = ReactionEncoder(
            vocab_size=max(len(tokenizer.vocab), 10),
            d_model=enc_cfg["d_model"],
            n_layers=enc_cfg["n_layers"],
            n_heads=enc_cfg["n_heads"],
            d_ff=enc_cfg["d_ff"],
            dropout=enc_cfg["dropout"],
            max_seq_len=enc_cfg["max_seq_len"],
        )

        # Load pretrained encoder if available
        pretrained_path = Path(__file__).parent.parent / "checkpoints" / "reaction_encoder_pretrain" / "best.pt"
        if pretrained_path.exists():
            load_checkpoint(encoder, pretrained_path, device)

        cond_embedder = ConditionEmbedder(
            discrete_vocab_sizes=discrete_vocab_sizes,
            n_continuous=cond_encoder.n_continuous,
            embed_dim=32,
            output_dim=128,
        )

        yield_model = YieldFeasibilityField(
            reaction_dim=enc_cfg["d_model"],
            condition_dim=128,
            hidden_dims=field_cfg["hidden_dims"],
            dropout=field_cfg["dropout"],
            gradient_penalty_weight=yield_cfg["robustness"]["gradient_penalty"],
        )

        bundle = YieldFieldBundle(encoder, cond_embedder, yield_model)
        n_params = sum(p.numel() for p in bundle.parameters())
        print(f"Parameters: {n_params:,}")

        optimizer = torch.optim.AdamW(bundle.parameters(), lr=train_cfg["lr"], weight_decay=base_cfg["training"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        def wrapped_loss(model, batch):
            return yield_field_loss_fn(
                (model.encoder, model.cond_embedder, model.yield_model), batch
            )

        trainer = Trainer(
            model=bundle,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=Path(__file__).parent.parent / "checkpoints",
            experiment_name=f"yield_field_{args.dataset}_{args.split}_member{member_idx}",
            patience=train_cfg["patience"],
            gradient_clip=base_cfg["training"]["gradient_clip"],
        )

        history = trainer.fit(train_loader, val_loader, wrapped_loss, max_epochs=train_cfg["max_epochs"])

    print(f"\nYield field ensemble training complete ({args.n_ensemble} members).")


if __name__ == "__main__":
    main()
