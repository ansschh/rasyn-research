"""
Training utilities: training loop, checkpointing, logging, reproducibility.
"""

import os
import random
import time
from pathlib import Path
from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EarlyStopping:
    """Early stopping based on validation metric."""

    def __init__(self, patience: int = 15, mode: str = "min", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        if self.best is None:
            self.best = metric
            return False

        if self.mode == "min":
            improved = metric < (self.best - self.min_delta)
        else:
            improved = metric > (self.best + self.min_delta)

        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict:
    """Load model checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return ckpt


class Trainer:
    """
    Generic training loop with validation, early stopping, and logging.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[Path] = None,
        experiment_name: str = "experiment",
        patience: int = 15,
        gradient_clip: float = 1.0,
        use_wandb: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or get_device()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.experiment_name = experiment_name
        self.gradient_clip = gradient_clip
        self.use_wandb = use_wandb

        self.early_stopping = EarlyStopping(patience=patience, mode="min")

        self.model.to(self.device)

        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                self.use_wandb = False

    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
    ) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        all_losses = {}
        n_batches = 0

        for batch in train_loader:
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            self.optimizer.zero_grad()
            losses = loss_fn(self.model, batch)

            loss = losses["total"]
            loss.backward()

            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()

            total_loss += loss.item()
            for k, v in losses.items():
                all_losses[k] = all_losses.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in all_losses.items()}

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        loss_fn: Callable,
    ) -> dict[str, float]:
        """Validate on held-out data."""
        self.model.eval()
        all_losses = {}
        n_batches = 0

        for batch in val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            losses = loss_fn(self.model, batch)

            for k, v in losses.items():
                all_losses[k] = all_losses.get(k, 0.0) + v.item()
            n_batches += 1

        return {k: v / max(n_batches, 1) for k, v in all_losses.items()}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        max_epochs: int = 100,
    ) -> dict:
        """Full training loop with validation and early stopping."""
        history = {"train": [], "val": []}
        best_val_loss = float("inf")

        for epoch in range(max_epochs):
            t0 = time.time()

            train_metrics = self.train_epoch(train_loader, loss_fn)
            val_metrics = self.validate(val_loader, loss_fn)

            if self.scheduler:
                self.scheduler.step(val_metrics.get("total", 0))

            elapsed = time.time() - t0
            history["train"].append(train_metrics)
            history["val"].append(val_metrics)

            # Logging
            print(f"Epoch {epoch+1:3d}/{max_epochs} ({elapsed:.1f}s) | "
                  f"train_loss={train_metrics['total']:.4f} | "
                  f"val_loss={val_metrics['total']:.4f}")

            if self.use_wandb:
                self.wandb.log({
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                    **{f"val/{k}": v for k, v in val_metrics.items()},
                    "epoch": epoch + 1,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

            # Checkpointing
            if val_metrics["total"] < best_val_loss:
                best_val_loss = val_metrics["total"]
                save_checkpoint(
                    self.model, self.optimizer, epoch,
                    val_metrics,
                    self.checkpoint_dir / self.experiment_name / "best.pt",
                )

            # Early stopping
            if self.early_stopping(val_metrics["total"]):
                print(f"Early stopping at epoch {epoch+1}")
                break

        return history
