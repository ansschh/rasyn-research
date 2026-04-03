"""
Train condition prediction models: point baseline vs condition manifold (EBM).

This is the main comparison for Experiment 2 (H1):
Does set-valued condition prediction beat point condition prediction?
"""

import sys
import json
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.featurization import SmilesTokenizer, ConditionEncoder, ReactionConditionDataset, create_dataloader
from data.preprocessing import load_records
from data.splits import get_splits
from models.reaction_encoder import ReactionEncoder
from models.condition_point import PointConditionPredictor
from models.condition_manifold import ConditionManifoldEBM, generate_negative_samples
from training.utils import set_seed, get_device, Trainer, load_checkpoint


# ─── Loss functions ───────────────────────────────────────────────────────────

def point_condition_loss_fn(model_bundle, batch):
    """Loss for point condition predictor (baseline)."""
    encoder, predictor = model_bundle

    # Encode reaction
    enc_out = encoder(batch["input_ids"], batch["attention_mask"])
    reaction_repr = enc_out["pooled"]

    # Predict conditions
    preds = predictor(reaction_repr)

    # Compute loss
    losses = predictor.compute_loss(
        preds,
        batch["discrete_conditions"],
        batch["continuous_conditions"],
        batch["continuous_mask"],
    )
    return losses


_discrete_vocab_size_list = None  # set in main() before training


def ebm_condition_loss_fn(model_bundle, batch):
    """Loss for energy-based condition manifold model."""
    encoder, ebm = model_bundle

    # Encode reaction
    enc_out = encoder(batch["input_ids"], batch["attention_mask"])
    reaction_repr = enc_out["pooled"]

    # Generate negatives with proper vocab sizes
    neg_discrete, neg_continuous = generate_negative_samples(
        batch["discrete_conditions"],
        batch["continuous_conditions"],
        n_negatives=64,
        noise_scale=0.1,
        discrete_vocab_sizes=_discrete_vocab_size_list,
    )

    # NCE loss
    losses = ebm.nce_loss(
        reaction_repr,
        batch["discrete_conditions"],
        batch["continuous_conditions"],
        neg_discrete.to(reaction_repr.device),
        neg_continuous.to(reaction_repr.device),
    )
    losses["total"] = losses["nce_loss"]
    return losses


# ─── Model bundle wrapper for Trainer ─────────────────────────────────────────

class ModelBundle(torch.nn.Module):
    """Wraps encoder + head for the Trainer."""
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["point", "ebm", "both"], default="both")
    parser.add_argument("--split", choices=["random", "scaffold", "source"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load configs
    base_cfg_path = Path(__file__).parent.parent / "config" / "base.yaml"
    cond_cfg_path = Path(__file__).parent.parent / "config" / "condition_manifold.yaml"
    with open(base_cfg_path) as f:
        base_cfg = yaml.safe_load(f)
    with open(cond_cfg_path) as f:
        cond_cfg = yaml.safe_load(f)

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    # Load data
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    df = load_records(processed_dir / "orderly.parquet")
    print(f"Loaded {len(df)} reactions")

    # Load tokenizer and condition encoder
    tokenizer = SmilesTokenizer.load(processed_dir / "smiles_vocab.json")
    with open(processed_dir / "condition_vocab.json") as f:
        discrete_vocabs = json.load(f)
    with open(processed_dir / "continuous_stats.json") as f:
        continuous_stats = json.load(f)
    cond_encoder = ConditionEncoder(discrete_vocabs, continuous_stats)

    # Split
    splits = get_splits(df, strategy=args.split, seed=args.seed)
    enc_cfg = base_cfg["encoder"]
    train_cfg = cond_cfg.get("training", base_cfg["training"])

    # Create datasets
    train_ds = ReactionConditionDataset(splits["train"], tokenizer, cond_encoder, max_seq_len=enc_cfg["max_seq_len"])
    val_ds = ReactionConditionDataset(splits["val"], tokenizer, cond_encoder, max_seq_len=enc_cfg["max_seq_len"])
    test_ds = ReactionConditionDataset(splits["test"], tokenizer, cond_encoder, max_seq_len=enc_cfg["max_seq_len"])

    train_loader = create_dataloader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False)

    # Load pretrained encoder
    encoder = ReactionEncoder(
        vocab_size=len(tokenizer.vocab),
        d_model=enc_cfg["d_model"],
        n_layers=enc_cfg["n_layers"],
        n_heads=enc_cfg["n_heads"],
        d_ff=enc_cfg["d_ff"],
        dropout=enc_cfg["dropout"],
        max_seq_len=enc_cfg["max_seq_len"],
    )
    pretrained_path = Path(__file__).parent.parent / "checkpoints" / "reaction_encoder_pretrain" / "best.pt"
    if pretrained_path.exists():
        load_checkpoint(encoder, pretrained_path, device)
        print("Loaded pretrained encoder")
    else:
        print("WARNING: No pretrained encoder found, training from scratch")

    # Discrete vocab sizes for heads
    discrete_vocab_sizes = {k: len(v) for k, v in discrete_vocabs.items()}

    # Set global for EBM negative sampling
    global _discrete_vocab_size_list
    _discrete_vocab_size_list = [len(v) for v in discrete_vocabs.values()]

    def train_model(name, loss_fn, model_bundle):
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")

        n_params = sum(p.numel() for p in model_bundle.parameters())
        print(f"Parameters: {n_params:,}")

        optimizer = torch.optim.AdamW(model_bundle.parameters(), lr=train_cfg["lr"], weight_decay=base_cfg["training"]["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        # Wrap loss_fn to unpack model_bundle
        def wrapped_loss(model, batch):
            return loss_fn((model.encoder, model.head), batch)

        trainer = Trainer(
            model=model_bundle,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=Path(__file__).parent.parent / "checkpoints",
            experiment_name=f"condition_{name}_{args.split}_seed{args.seed}",
            patience=train_cfg["patience"],
            gradient_clip=base_cfg["training"]["gradient_clip"],
        )

        history = trainer.fit(train_loader, val_loader, wrapped_loss, max_epochs=train_cfg["max_epochs"])
        return history

    # ─── Train point baseline ────────────────────────────────────────────
    if args.model in ("point", "both"):
        point_head = PointConditionPredictor(
            reaction_dim=enc_cfg["d_model"],
            discrete_vocab_sizes=discrete_vocab_sizes,
            n_continuous=cond_encoder.n_continuous,
        )
        point_bundle = ModelBundle(encoder, point_head)
        train_model("point", point_condition_loss_fn, point_bundle)

    # ─── Train EBM manifold ─────────────────────────────────────────────
    if args.model in ("ebm", "both"):
        # Fresh encoder copy for EBM (so it trains independently)
        encoder_ebm = ReactionEncoder(
            vocab_size=len(tokenizer.vocab),
            d_model=enc_cfg["d_model"],
            n_layers=enc_cfg["n_layers"],
            n_heads=enc_cfg["n_heads"],
            d_ff=enc_cfg["d_ff"],
            dropout=enc_cfg["dropout"],
            max_seq_len=enc_cfg["max_seq_len"],
        )
        if pretrained_path.exists():
            load_checkpoint(encoder_ebm, pretrained_path, device)

        ebm_head = ConditionManifoldEBM(
            reaction_dim=enc_cfg["d_model"],
            condition_dim=cond_cfg["ebm"]["condition_dim"],
            hidden_dims=cond_cfg["ebm"]["energy_hidden"],
            discrete_vocab_sizes=discrete_vocab_sizes,
            n_continuous=cond_encoder.n_continuous,
        )
        ebm_bundle = ModelBundle(encoder_ebm, ebm_head)
        train_model("ebm", ebm_condition_loss_fn, ebm_bundle)

    print("\nCondition model training complete.")


if __name__ == "__main__":
    main()
