"""Tests for training utilities."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from adaptive_hierarchical_reconciliation_with_attention_pruning.models import (
    HierarchicalForecastModel,
    HierarchicalCoherenceLoss,
)
from adaptive_hierarchical_reconciliation_with_attention_pruning.training import (
    HierarchicalTrainer,
)


class TestHierarchicalTrainer:
    """Tests for HierarchicalTrainer."""

    @pytest.fixture
    def trainer_setup(self, device):
        """Setup trainer for testing."""
        model = HierarchicalForecastModel(
            input_dim=1,
            calendar_dim=3,
            hidden_dim=16,
            num_layers=1,
            forecast_horizon=3,
            num_bottom_level=10,
            num_total_series=20,
            num_attention_heads=2,
        )

        loss_fn = HierarchicalCoherenceLoss(
            coherence_weight=0.3, forecast_weight=0.7, use_rmsse=False
        )

        aggregation_matrix = torch.randn(10, 20)

        trainer = HierarchicalTrainer(
            model=model,
            loss_fn=loss_fn,
            aggregation_matrix=aggregation_matrix,
            learning_rate=0.001,
            device=device,
            patience=5,
        )

        return trainer

    @pytest.fixture
    def sample_dataloader(self):
        """Create sample data loader."""
        X = torch.randn(20, 7, 1)
        y = torch.randn(20, 3)
        cal = torch.randn(20, 7, 3)

        dataset = TensorDataset(X, y, cal)
        loader = DataLoader(dataset, batch_size=4, shuffle=False)

        return loader

    def test_trainer_initialization(self, trainer_setup):
        """Test trainer initialization."""
        trainer = trainer_setup

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.patience == 5

    def test_train_epoch(self, trainer_setup, sample_dataloader):
        """Test training for one epoch."""
        trainer = trainer_setup

        metrics = trainer.train_epoch(sample_dataloader, epoch=0)

        assert "train_loss" in metrics
        assert "train_forecast_loss" in metrics
        assert "train_coherence_loss" in metrics
        assert metrics["train_loss"] > 0

    def test_validate(self, trainer_setup, sample_dataloader):
        """Test validation."""
        trainer = trainer_setup

        metrics = trainer.validate(sample_dataloader)

        assert "val_loss" in metrics
        assert "val_forecast_loss" in metrics
        assert "val_coherence_loss" in metrics
        assert metrics["val_loss"] > 0

    def test_training_loop(self, trainer_setup, sample_dataloader):
        """Test full training loop."""
        trainer = trainer_setup

        history = trainer.train(
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            num_epochs=3,
            checkpoint_dir="checkpoints_test",
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) <= 3  # May stop early
        assert len(history["train_loss"]) == len(history["val_loss"])

    def test_early_stopping(self, trainer_setup, sample_dataloader):
        """Test early stopping mechanism."""
        trainer = trainer_setup
        trainer.patience = 2

        # Create validation loader with constant data (no improvement)
        X_val = torch.ones(20, 7, 1)
        y_val = torch.ones(20, 3)
        cal_val = torch.ones(20, 7, 3)
        val_dataset = TensorDataset(X_val, y_val, cal_val)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

        history = trainer.train(
            train_loader=sample_dataloader,
            val_loader=val_loader,
            num_epochs=20,
            checkpoint_dir="checkpoints_test",
        )

        # Should stop before 20 epochs due to early stopping
        assert len(history["train_loss"]) < 20

    def test_gradient_clipping(self, trainer_setup, sample_dataloader):
        """Test gradient clipping."""
        trainer = trainer_setup
        trainer.gradient_clip = 0.5

        # Training should not raise errors with gradient clipping
        metrics = trainer.train_epoch(sample_dataloader, epoch=0)

        assert metrics["train_loss"] > 0

    def test_learning_rate_scheduling(self, trainer_setup, sample_dataloader):
        """Test learning rate scheduling."""
        trainer = trainer_setup

        history = trainer.train(
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            num_epochs=5,
            checkpoint_dir="checkpoints_test",
        )

        # Learning rate should change over epochs
        lrs = history["learning_rate"]
        assert len(lrs) > 0
        # With cosine annealing, LR should decrease
        assert lrs[-1] < lrs[0]
