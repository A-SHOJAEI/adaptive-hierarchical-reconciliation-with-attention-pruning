"""Evaluation metrics for hierarchical forecasting."""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class RMSSEMetric:
    """Root Mean Scaled Squared Error metric for M5 competition."""

    def __init__(self, train_data: Optional[np.ndarray] = None):
        """Initialize RMSSE metric.

        Args:
            train_data: Training data for computing scale (num_series, num_timesteps)
        """
        self.scale = None
        if train_data is not None:
            self._compute_scale(train_data)

    def _compute_scale(self, train_data: np.ndarray) -> None:
        """Compute scale factors from training data.

        Args:
            train_data: Training data
        """
        # Compute naive forecast errors (seasonal naive with period=1)
        naive_errors = np.diff(train_data, axis=1) ** 2
        self.scale = np.sqrt(np.mean(naive_errors, axis=1, keepdims=True))
        logger.info(f"Computed RMSSE scale: {self.scale.shape}")

    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        scale: Optional[np.ndarray] = None,
    ) -> float:
        """Compute RMSSE.

        Args:
            predictions: Predictions (num_series, num_timesteps)
            targets: Ground truth (num_series, num_timesteps)
            scale: Optional scale factors

        Returns:
            RMSSE value
        """
        if scale is None:
            scale = self.scale

        if scale is None:
            # Fall back to RMSE if no scale available
            logger.warning("No scale available, using RMSE instead")
            return np.sqrt(np.mean((predictions - targets) ** 2))

        # Compute squared errors
        squared_errors = (predictions - targets) ** 2

        # Scale errors
        scaled_errors = squared_errors / (scale + 1e-8)

        # RMSSE
        rmsse = np.sqrt(np.mean(scaled_errors))

        return rmsse


class HierarchicalCoherenceMetric:
    """Measure hierarchical coherence of forecasts."""

    def __init__(self, aggregation_matrix: np.ndarray):
        """Initialize coherence metric.

        Args:
            aggregation_matrix: Summing matrix (num_bottom, num_total)
        """
        self.aggregation_matrix = aggregation_matrix
        logger.info(
            f"Initialized HierarchicalCoherenceMetric: {aggregation_matrix.shape}"
        )

    def compute(
        self, bottom_predictions: np.ndarray, all_predictions: np.ndarray
    ) -> float:
        """Compute hierarchical coherence score.

        Args:
            bottom_predictions: Bottom-level predictions (num_bottom, horizon)
            all_predictions: All levels predictions (num_total, horizon)

        Returns:
            Coherence score (1.0 = perfect coherence, 0.0 = no coherence)
        """
        # Aggregate bottom-level predictions using summing matrix
        aggregated = np.matmul(
            self.aggregation_matrix.T, bottom_predictions
        )  # (num_total, horizon)

        # Compute relative error
        denominator = np.abs(all_predictions) + 1e-8
        relative_errors = np.abs(all_predictions - aggregated) / denominator

        # Coherence score: 1 - mean relative error
        coherence = 1.0 - np.mean(relative_errors)

        return max(0.0, coherence)  # Clamp to [0, 1]


class PruningRatioMetric:
    """Measure the pruning ratio of hierarchical constraints."""

    def compute(self, pruning_mask: np.ndarray) -> float:
        """Compute pruning ratio.

        Args:
            pruning_mask: Binary mask (num_bottom, num_total)

        Returns:
            Fraction of connections pruned
        """
        if pruning_mask is None:
            return 0.0

        # Count pruned connections (mask value < 0.5)
        pruned = (pruning_mask < 0.5).astype(float).mean()

        return pruned


class ForecastMetrics:
    """Collection of forecasting metrics."""

    @staticmethod
    def mae(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Mean Absolute Error.

        Args:
            predictions: Predictions
            targets: Ground truth

        Returns:
            MAE value
        """
        return np.mean(np.abs(predictions - targets))

    @staticmethod
    def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Root Mean Squared Error.

        Args:
            predictions: Predictions
            targets: Ground truth

        Returns:
            RMSE value
        """
        return np.sqrt(np.mean((predictions - targets) ** 2))

    @staticmethod
    def mape(
        predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-8
    ) -> float:
        """Mean Absolute Percentage Error.

        Args:
            predictions: Predictions
            targets: Ground truth
            epsilon: Small constant to avoid division by zero

        Returns:
            MAPE value (in percentage)
        """
        return np.mean(
            np.abs((targets - predictions) / (targets + epsilon))
        ) * 100

    @staticmethod
    def smape(
        predictions: np.ndarray, targets: np.ndarray, epsilon: float = 1e-8
    ) -> float:
        """Symmetric Mean Absolute Percentage Error.

        Args:
            predictions: Predictions
            targets: Ground truth
            epsilon: Small constant to avoid division by zero

        Returns:
            SMAPE value (in percentage)
        """
        numerator = np.abs(targets - predictions)
        denominator = (np.abs(targets) + np.abs(predictions)) / 2 + epsilon
        return np.mean(numerator / denominator) * 100
