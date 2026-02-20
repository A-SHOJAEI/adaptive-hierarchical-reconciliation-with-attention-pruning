"""Training utilities for hierarchical forecasting."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class HierarchicalTrainer:
    """Trainer for hierarchical forecasting with early stopping and LR scheduling."""

    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        aggregation_matrix: torch.Tensor,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        device: torch.device = None,
        gradient_clip: float = 1.0,
        patience: int = 10,
        min_delta: float = 0.0001,
    ):
        """Initialize trainer.

        Args:
            model: Model to train
            loss_fn: Loss function
            aggregation_matrix: Hierarchical aggregation matrix
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to train on
            gradient_clip: Gradient clipping threshold
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
        """
        self.model = model
        self.loss_fn = loss_fn
        self.aggregation_matrix = aggregation_matrix
        self.device = (
            device if device is not None else torch.device("cpu")
        )
        self.gradient_clip = gradient_clip
        self.patience = patience
        self.min_delta = min_delta

        # Move model and aggregation matrix to device
        self.model.to(self.device)
        self.aggregation_matrix = self.aggregation_matrix.to(self.device)

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = None  # Will be set in train()

        # Early stopping state
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.best_model_state = None

        logger.info(
            f"Initialized HierarchicalTrainer on device: {self.device}"
        )

    def train_epoch(
        self, train_loader: DataLoader, epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            train_loader: Training data loader
            epoch: Current epoch number

        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_forecast_loss = 0.0
        total_coherence_loss = 0.0
        num_batches = 0

        for batch_idx, (x, y, cal_features) in enumerate(train_loader):
            # Move to device
            x = x.to(self.device)
            y = y.to(self.device)
            cal_features = cal_features.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            outputs = self.model(
                x, cal_features, self.aggregation_matrix, hard_pruning=False
            )

            # Compute loss
            loss_dict = self.loss_fn(
                predictions=outputs["reconciled_predictions"],
                targets=y,
                aggregation_matrix=self.aggregation_matrix,
            )

            total_loss_batch = loss_dict["total_loss"]

            # Backward pass
            total_loss_batch.backward()

            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_forecast_loss += loss_dict["forecast_loss"].item()
            total_coherence_loss += loss_dict["coherence_loss"].item()
            num_batches += 1

            if batch_idx % 10 == 0:
                logger.debug(
                    f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                    f"Loss: {total_loss_batch.item():.4f}"
                )

        # Average losses
        metrics = {
            "train_loss": total_loss / num_batches,
            "train_forecast_loss": total_forecast_loss / num_batches,
            "train_coherence_loss": total_coherence_loss / num_batches,
        }

        return metrics

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model.

        Args:
            val_loader: Validation data loader

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_forecast_loss = 0.0
        total_coherence_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, y, cal_features in val_loader:
                # Move to device
                x = x.to(self.device)
                y = y.to(self.device)
                cal_features = cal_features.to(self.device)

                # Forward pass
                outputs = self.model(
                    x, cal_features, self.aggregation_matrix, hard_pruning=True
                )

                # Compute loss
                loss_dict = self.loss_fn(
                    predictions=outputs["reconciled_predictions"],
                    targets=y,
                    aggregation_matrix=self.aggregation_matrix,
                )

                total_loss += loss_dict["total_loss"].item()
                total_forecast_loss += loss_dict["forecast_loss"].item()
                total_coherence_loss += loss_dict["coherence_loss"].item()
                num_batches += 1

        # Average losses
        metrics = {
            "val_loss": total_loss / num_batches,
            "val_forecast_loss": total_forecast_loss / num_batches,
            "val_coherence_loss": total_coherence_loss / num_batches,
        }

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        checkpoint_dir: str = "checkpoints",
    ) -> Dict[str, list]:
        """Full training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            checkpoint_dir: Directory to save checkpoints

        Returns:
            Dictionary of training history
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs)

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_forecast_loss": [],
            "val_forecast_loss": [],
            "train_coherence_loss": [],
            "val_coherence_loss": [],
            "learning_rate": [],
        }

        logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record history
            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            history["train_forecast_loss"].append(
                train_metrics["train_forecast_loss"]
            )
            history["val_forecast_loss"].append(
                val_metrics["val_forecast_loss"]
            )
            history["train_coherence_loss"].append(
                train_metrics["train_coherence_loss"]
            )
            history["val_coherence_loss"].append(
                val_metrics["val_coherence_loss"]
            )
            history["learning_rate"].append(current_lr)

            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"LR: {current_lr:.6f}"
            )

            # Early stopping check
            if val_metrics["val_loss"] < self.best_loss - self.min_delta:
                self.best_loss = val_metrics["val_loss"]
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()

                # Save best model
                save_path = checkpoint_path / "best_model.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_metrics["val_loss"],
                        "aggregation_matrix": self.aggregation_matrix.cpu(),
                    },
                    save_path,
                )
                logger.info(f"Saved best model to {save_path}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement. Patience: {self.patience_counter}/{self.patience}"
                )

                if self.patience_counter >= self.patience:
                    logger.info(
                        f"Early stopping triggered at epoch {epoch+1}"
                    )
                    break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("Restored best model weights")

        return history
