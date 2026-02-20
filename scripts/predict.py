#!/usr/bin/env python
"""Prediction script for hierarchical forecasting model."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

from adaptive_hierarchical_reconciliation_with_attention_pruning.models import (
    HierarchicalForecastModel,
)
from adaptive_hierarchical_reconciliation_with_attention_pruning.utils import (
    load_config,
    setup_logging,
    get_device,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Make predictions with hierarchical forecasting model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input data (numpy array: seq_len x features)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.json",
        help="Output file for predictions",
    )
    parser.add_argument(
        "--calendar-features",
        type=str,
        default=None,
        help="Path to calendar features (numpy array)",
    )
    return parser.parse_args()


def load_model(checkpoint_path, config, device):
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        config: Configuration dictionary
        device: Device to load model on

    Returns:
        Tuple of (model, aggregation_matrix)
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Load checkpoint first to get dimensions
    checkpoint = torch.load(checkpoint_path, map_location=device)
    aggregation_matrix = checkpoint["aggregation_matrix"]

    num_bottom_level = aggregation_matrix.shape[0]
    num_total_series = aggregation_matrix.shape[1]

    # Create model
    model = HierarchicalForecastModel(
        input_dim=config.get("input_dim", 1),
        calendar_dim=config.get("calendar_dim", 3),
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 2),
        forecast_horizon=config.get("forecast_horizon", 7),
        num_bottom_level=num_bottom_level,
        num_total_series=num_total_series,
        num_attention_heads=config.get("num_attention_heads", 4),
        dropout=config.get("dropout", 0.1),
        enable_pruning=config.get("enable_pruning", True),
        target_pruning_ratio=config.get("target_pruning_ratio", 0.4),
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model, aggregation_matrix


def load_input_data(input_path, calendar_path, config):
    """Load input data for prediction.

    Args:
        input_path: Path to input sales data
        calendar_path: Path to calendar features
        config: Configuration dictionary

    Returns:
        Tuple of (input_tensor, calendar_tensor)
    """
    logger.info(f"Loading input data from {input_path}")

    # Load input data
    input_data = np.load(input_path)

    # Ensure correct shape (seq_len, features)
    if input_data.ndim == 1:
        input_data = input_data.reshape(-1, 1)

    # Check sequence length
    expected_len = config.get("lookback_window", 28)
    if input_data.shape[0] != expected_len:
        logger.warning(
            f"Input sequence length {input_data.shape[0]} != expected {expected_len}"
        )
        if input_data.shape[0] > expected_len:
            input_data = input_data[-expected_len:]
        else:
            # Pad with zeros
            padding = np.zeros((expected_len - input_data.shape[0], input_data.shape[1]))
            input_data = np.vstack([padding, input_data])

    # Load or create calendar features
    if calendar_path:
        calendar_features = np.load(calendar_path)
    else:
        # Create dummy calendar features
        calendar_dim = config.get("calendar_dim", 3)
        calendar_features = np.zeros((input_data.shape[0], calendar_dim))
        logger.warning("No calendar features provided, using zeros")

    # Convert to tensors
    input_tensor = torch.FloatTensor(input_data).unsqueeze(0)  # Add batch dimension
    calendar_tensor = torch.FloatTensor(calendar_features).unsqueeze(0)

    return input_tensor, calendar_tensor


def make_predictions(model, input_data, calendar_features, aggregation_matrix, device):
    """Make predictions.

    Args:
        model: Trained model
        input_data: Input tensor (1, seq_len, features)
        calendar_features: Calendar tensor (1, seq_len, cal_features)
        aggregation_matrix: Aggregation matrix
        device: Device

    Returns:
        Dictionary with predictions and confidence scores
    """
    logger.info("Making predictions...")

    with torch.no_grad():
        input_data = input_data.to(device)
        calendar_features = calendar_features.to(device)
        aggregation_matrix = aggregation_matrix.to(device)

        outputs = model(
            input_data, calendar_features, aggregation_matrix, hard_pruning=True
        )

        bottom_predictions = outputs["bottom_predictions"].cpu().numpy()
        reconciled_predictions = outputs["reconciled_predictions"].cpu().numpy()
        attention_weights = outputs["attention_weights"].cpu().numpy()

    # Compute confidence scores (using attention entropy)
    # Lower entropy = higher confidence
    epsilon = 1e-8
    attention_probs = attention_weights[0]  # (horizon, num_total, num_bottom)
    entropy = -np.sum(
        attention_probs * np.log(attention_probs + epsilon), axis=-1
    )  # (horizon, num_total)
    confidence = 1.0 - (entropy / np.log(attention_probs.shape[-1]))  # Normalize

    results = {
        "bottom_predictions": bottom_predictions[0].tolist(),
        "reconciled_predictions": reconciled_predictions[0].tolist(),
        "confidence_scores": confidence.mean(axis=0).tolist(),  # Average across horizon
        "forecast_horizon": bottom_predictions.shape[1],
    }

    return results


def main():
    """Main prediction function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO")
    logger.info("Starting prediction pipeline")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Get device
        device = get_device()
        logger.info(f"Using device: {device}")

        # Load model
        model, aggregation_matrix = load_model(args.checkpoint, config, device)

        # Load input data
        input_data, calendar_features = load_input_data(
            args.input, args.calendar_features, config
        )

        # Make predictions
        results = make_predictions(
            model, input_data, calendar_features, aggregation_matrix, device
        )

        # Save predictions
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Predictions saved to {output_path}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Prediction Summary")
        logger.info("=" * 60)
        logger.info(f"Forecast horizon: {results['forecast_horizon']}")
        logger.info(f"Bottom-level predictions: {len(results['bottom_predictions'])}")
        logger.info(
            f"Reconciled predictions: {len(results['reconciled_predictions'])}"
        )
        logger.info(
            f"Average confidence: {np.mean(results['confidence_scores']):.4f}"
        )
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
