"""
Pretrain the shared Reaction Encoder via Masked Token Modeling.

Masks ~15% of SMILES tokens and trains the encoder to reconstruct them.
This gives the encoder a general understanding of reaction SMILES structure
before fine-tuning for condition or yield prediction.
"""

import sys
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.featurization import SmilesTokenizer, ReactionConditionDataset, create_dataloader
from data.preprocessing import load_records
from data.splits import random_split
from models.reaction_encoder import ReactionEncoder
from training.utils import set_seed, get_device, Trainer


MASK_PROB = 0.15


def mlm_loss_fn(model: ReactionEncoder, batch: dict) -> dict:
    """Masked Language Model loss."""
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    # Create masked version
    mask_prob = torch.rand_like(input_ids.float()) < MASK_PROB
    mask_prob = mask_prob & (attention_mask > 0)  # don't mask padding
    mask_prob[:, 0] = False  # don't mask BOS
    mask_prob[:, -1] = False  # don't mask EOS

    labels = input_ids.clone()
    labels[~mask_prob] = -100  # ignore unmasked tokens

    masked_ids = input_ids.clone()
    mask_token_id = 4  # <MASK>
    # 80% replace with MASK, 10% random, 10% keep
    replace_mask = mask_prob & (torch.rand_like(input_ids.float()) < 0.8)
    random_mask = mask_prob & ~replace_mask & (torch.rand_like(input_ids.float()) < 0.5)

    masked_ids[replace_mask] = mask_token_id
    masked_ids[random_mask] = torch.randint_like(masked_ids[random_mask], 6, model.token_embedding.num_embeddings)

    logits = model.mlm_forward(masked_ids, attention_mask)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)

    # Token accuracy on masked positions
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        correct = (pred == input_ids) & mask_prob
        acc = correct.sum().float() / mask_prob.sum().float().clamp(min=1)

    return {"total": loss, "mlm_loss": loss, "token_accuracy": acc}


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "base.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # Load data
    processed_dir = Path(__file__).parent.parent / "data" / "processed"
    df = load_records(processed_dir / "orderly.parquet")
    print(f"Loaded {len(df)} reactions")

    # Build tokenizer
    tokenizer = SmilesTokenizer()
    all_smiles = df["reaction_smiles"].dropna().tolist()
    tokenizer.build_vocab(all_smiles)
    tokenizer.save(processed_dir / "smiles_vocab.json")
    print(f"Vocab size: {len(tokenizer.vocab)}")

    # Split
    splits = random_split(df, test_frac=0.1, val_frac=0.1, seed=seed)

    # We need a minimal condition encoder for the dataset (even though we're only doing MLM)
    import json
    vocab_path = processed_dir / "condition_vocab.json"
    stats_path = processed_dir / "continuous_stats.json"

    if vocab_path.exists() and stats_path.exists():
        with open(vocab_path) as f:
            discrete_vocabs = json.load(f)
        with open(stats_path) as f:
            continuous_stats = json.load(f)
    else:
        # Minimal fallback
        discrete_vocabs = {k: {"<PAD>": 0, "<UNK>": 1, "<NONE>": 2} for k in ["solvent", "catalyst", "reagent", "base", "ligand"]}
        continuous_stats = {"temperature": {"mean": 25, "std": 50, "min": -78, "max": 300}}

    from data.featurization import ConditionEncoder
    cond_encoder = ConditionEncoder(discrete_vocabs, continuous_stats)

    # Create datasets
    enc_config = config["encoder"]
    train_ds = ReactionConditionDataset(splits["train"], tokenizer, cond_encoder, max_seq_len=enc_config["max_seq_len"])
    val_ds = ReactionConditionDataset(splits["val"], tokenizer, cond_encoder, max_seq_len=enc_config["max_seq_len"])

    train_cfg = config["training"]
    train_loader = create_dataloader(train_ds, batch_size=train_cfg["batch_size"], shuffle=True, num_workers=train_cfg["num_workers"])
    val_loader = create_dataloader(val_ds, batch_size=train_cfg["batch_size"], shuffle=False, num_workers=train_cfg["num_workers"])

    # Model
    model = ReactionEncoder(
        vocab_size=len(tokenizer.vocab),
        d_model=enc_config["d_model"],
        n_layers=enc_config["n_layers"],
        n_heads=enc_config["n_heads"],
        d_ff=enc_config["d_ff"],
        dropout=enc_config["dropout"],
        max_seq_len=enc_config["max_seq_len"],
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Reaction Encoder: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # Train
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=Path(__file__).parent.parent / "checkpoints",
        experiment_name="reaction_encoder_pretrain",
        patience=train_cfg["patience"],
        gradient_clip=train_cfg["gradient_clip"],
    )

    history = trainer.fit(train_loader, val_loader, mlm_loss_fn, max_epochs=train_cfg["max_epochs"])
    print("Encoder pretraining complete.")


if __name__ == "__main__":
    main()
