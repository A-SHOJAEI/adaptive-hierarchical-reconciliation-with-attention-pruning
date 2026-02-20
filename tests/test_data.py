"""Tests for data loading and preprocessing."""

import numpy as np
import pytest
import torch

from adaptive_hierarchical_reconciliation_with_attention_pruning.data import (
    M5DataLoader,
    M5Preprocessor,
    HierarchyBuilder,
)


class TestM5DataLoader:
    """Tests for M5DataLoader."""

    def test_load_or_generate(self):
        """Test data loading/generation."""
        loader = M5DataLoader(data_dir="data/test")
        sales_df, calendar_df, hierarchy_df = loader.load_or_generate(
            num_items=10, num_days=100
        )

        assert sales_df.shape[0] == 10
        assert len([col for col in sales_df.columns if col.startswith("d_")]) == 100
        assert calendar_df.shape[0] == 100
        assert hierarchy_df.shape[0] == 10

    def test_hierarchy_structure(self):
        """Test hierarchy structure."""
        loader = M5DataLoader(data_dir="data/test")
        sales_df, calendar_df, hierarchy_df = loader.load_or_generate(
            num_items=18, num_days=50
        )

        # Check hierarchy columns
        assert "item_id" in hierarchy_df.columns
        assert "state_id" in hierarchy_df.columns
        assert "store_id" in hierarchy_df.columns
        assert "cat_id" in hierarchy_df.columns


class TestHierarchyBuilder:
    """Tests for HierarchyBuilder."""

    def test_build_aggregation_matrix(self, sample_hierarchy_data):
        """Test aggregation matrix construction."""
        builder = HierarchyBuilder(sample_hierarchy_data)
        S = builder.build_aggregation_matrix()

        assert S.shape[0] == len(sample_hierarchy_data)
        assert S.shape[1] > S.shape[0]  # Should have more total series
        assert np.all(S >= 0)  # All elements should be non-negative

    def test_level_indices(self, sample_hierarchy_data):
        """Test level indices mapping."""
        builder = HierarchyBuilder(sample_hierarchy_data)
        builder.build_aggregation_matrix()

        assert "total" in builder.level_indices
        assert "state" in builder.level_indices
        assert "store" in builder.level_indices
        assert "category" in builder.level_indices
        assert "item" in builder.level_indices


class TestM5Preprocessor:
    """Tests for M5Preprocessor."""

    def test_prepare_sequences(self, sample_sales_data, sample_calendar_data):
        """Test sequence preparation."""
        preprocessor = M5Preprocessor(
            lookback_window=7, forecast_horizon=3, detect_changepoints=False
        )

        X, y, cal_features = preprocessor.prepare_sequences(
            sample_sales_data, sample_calendar_data
        )

        assert X.shape[1] == 7  # Lookback window
        assert X.shape[2] == 1  # Input dimension
        assert y.shape[1] == 3  # Forecast horizon
        assert cal_features.shape[1] == 7  # Lookback window
        assert cal_features.shape[2] == 3  # Calendar features

    def test_sequence_shapes(self, sample_sales_data, sample_calendar_data):
        """Test output shapes are correct."""
        preprocessor = M5Preprocessor(
            lookback_window=14, forecast_horizon=7, detect_changepoints=False
        )

        X, y, cal_features = preprocessor.prepare_sequences(
            sample_sales_data, sample_calendar_data
        )

        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert isinstance(cal_features, torch.Tensor)
        assert X.shape[0] == y.shape[0]  # Same number of samples

    def test_inverse_transform(self, sample_sales_data, sample_calendar_data):
        """Test inverse transformation."""
        preprocessor = M5Preprocessor(lookback_window=7, forecast_horizon=3)

        X, y, cal_features = preprocessor.prepare_sequences(
            sample_sales_data, sample_calendar_data
        )

        # Create dummy predictions
        predictions = np.random.randn(10, 3)

        # Inverse transform should not raise errors
        restored = preprocessor.inverse_transform(predictions.T)
        assert restored.shape == (3, 10)
