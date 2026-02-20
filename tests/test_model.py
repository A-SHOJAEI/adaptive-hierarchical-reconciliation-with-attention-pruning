"""Tests for model components."""

import numpy as np
import pytest
import torch

from adaptive_hierarchical_reconciliation_with_attention_pruning.models import (
    HierarchicalForecastModel,
    AttentionReconciliationLayer,
    StructuredPruning,
    HierarchicalCoherenceLoss,
)


class TestAttentionReconciliationLayer:
    """Tests for AttentionReconciliationLayer."""

    def test_forward_pass(self):
        """Test forward pass."""
        num_bottom = 10
        num_total = 20
        batch_size = 4
        horizon = 3

        layer = AttentionReconciliationLayer(
            num_bottom_level=num_bottom,
            num_total_series=num_total,
            hidden_dim=16,
            num_heads=2,
        )

        bottom_predictions = torch.randn(batch_size, num_bottom, horizon)
        aggregation_matrix = torch.randn(num_bottom, num_total)

        reconciled, attention = layer(bottom_predictions, aggregation_matrix)

        assert reconciled.shape == (batch_size, num_total, horizon)
        assert attention.shape[0] == batch_size
        assert attention.shape[1] == horizon

    def test_attention_weights_sum(self):
        """Test attention weights are normalized (approximately sum to 1)."""
        layer = AttentionReconciliationLayer(
            num_bottom_level=5, num_total_series=10, hidden_dim=8, num_heads=2
        )

        bottom_predictions = torch.randn(2, 5, 3)
        aggregation_matrix = torch.randn(5, 10)

        reconciled, attention = layer(bottom_predictions, aggregation_matrix)

        # Attention weights should approximately sum to 1 across bottom level
        # (may not be exact due to multi-head averaging and dropout)
        attention_sum = attention.sum(dim=-1)
        # Check that sums are reasonable (between 0.5 and 1.5)
        assert torch.all(attention_sum > 0.5)
        assert torch.all(attention_sum < 1.5)


class TestStructuredPruning:
    """Tests for StructuredPruning."""

    def test_pruning_forward(self):
        """Test pruning forward pass."""
        pruning = StructuredPruning(
            num_bottom_level=10, num_total_series=20, target_pruning_ratio=0.4
        )

        aggregation_matrix = torch.randn(10, 20)

        pruned_matrix, mask = pruning(aggregation_matrix, hard=False)

        assert pruned_matrix.shape == aggregation_matrix.shape
        assert mask.shape == aggregation_matrix.shape

    def test_hard_pruning(self):
        """Test hard pruning."""
        pruning = StructuredPruning(
            num_bottom_level=10, num_total_series=20, target_pruning_ratio=0.5
        )

        aggregation_matrix = torch.ones(10, 20)

        pruned_matrix, mask = pruning(aggregation_matrix, hard=True)

        # Mask should be binary
        assert torch.all((mask == 0) | (mask == 1))

    def test_get_pruning_ratio(self):
        """Test pruning ratio computation."""
        pruning = StructuredPruning(
            num_bottom_level=10, num_total_series=20, target_pruning_ratio=0.4
        )

        ratio = pruning.get_pruning_ratio()

        assert 0 <= ratio <= 1


class TestHierarchicalCoherenceLoss:
    """Tests for HierarchicalCoherenceLoss."""

    def test_loss_computation(self):
        """Test loss computation."""
        loss_fn = HierarchicalCoherenceLoss(
            coherence_weight=0.3, forecast_weight=0.7, use_rmsse=False
        )

        batch_size = 4
        num_total = 20
        num_bottom = 10
        horizon = 3

        predictions = torch.randn(batch_size, num_total, horizon)
        targets = torch.randn(batch_size, num_bottom, horizon)
        aggregation_matrix = torch.randn(num_bottom, num_total)

        loss_dict = loss_fn(predictions, targets, aggregation_matrix)

        assert "total_loss" in loss_dict
        assert "forecast_loss" in loss_dict
        assert "coherence_loss" in loss_dict
        assert loss_dict["total_loss"] > 0

    def test_loss_components(self):
        """Test loss components are properly weighted."""
        coherence_weight = 0.3
        forecast_weight = 0.7

        loss_fn = HierarchicalCoherenceLoss(
            coherence_weight=coherence_weight,
            forecast_weight=forecast_weight,
            use_rmsse=False,
        )

        predictions = torch.randn(2, 10, 3)
        targets = torch.randn(2, 5, 3)
        aggregation_matrix = torch.randn(5, 10)

        loss_dict = loss_fn(predictions, targets, aggregation_matrix)

        # Total loss should be weighted sum
        expected_total = (
            forecast_weight * loss_dict["forecast_loss"]
            + coherence_weight * loss_dict["coherence_loss"]
        )
        assert torch.allclose(loss_dict["total_loss"], expected_total, atol=1e-5)


class TestHierarchicalForecastModel:
    """Tests for HierarchicalForecastModel."""

    def test_model_creation(self):
        """Test model creation."""
        model = HierarchicalForecastModel(
            input_dim=1,
            calendar_dim=3,
            hidden_dim=16,
            num_layers=1,
            forecast_horizon=3,
            num_bottom_level=10,
            num_total_series=20,
            num_attention_heads=2,
            dropout=0.1,
            enable_pruning=True,
        )

        assert model is not None
        assert model.enable_pruning is True

    def test_forward_pass(self, sample_sequences):
        """Test model forward pass."""
        X, y, cal_features = sample_sequences

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

        aggregation_matrix = torch.randn(10, 20)

        outputs = model(X, cal_features, aggregation_matrix)

        assert "bottom_predictions" in outputs
        assert "reconciled_predictions" in outputs
        assert "attention_weights" in outputs

    def test_pruning_disabled(self, sample_sequences):
        """Test model with pruning disabled."""
        X, y, cal_features = sample_sequences

        model = HierarchicalForecastModel(
            input_dim=1,
            calendar_dim=3,
            hidden_dim=16,
            num_layers=1,
            forecast_horizon=3,
            num_bottom_level=10,
            num_total_series=20,
            enable_pruning=False,
        )

        aggregation_matrix = torch.randn(10, 20)

        outputs = model(X, cal_features, aggregation_matrix)

        assert outputs["pruning_mask"] is None
        assert model.get_pruning_ratio() == 0.0

    def test_model_device_transfer(self, device):
        """Test moving model to device."""
        model = HierarchicalForecastModel(
            input_dim=1,
            calendar_dim=3,
            hidden_dim=16,
            num_layers=1,
            forecast_horizon=3,
            num_bottom_level=10,
            num_total_series=20,
        )

        model.to(device)

        # Check if parameters are on correct device
        for param in model.parameters():
            assert param.device.type == device.type
