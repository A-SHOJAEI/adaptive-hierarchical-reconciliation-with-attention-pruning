#!/usr/bin/env python
"""Evaluation script for hierarchical forecasting model."""

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
from torch.utils.data import DataLoader, TensorDataset

from adaptive_hierarchical_reconciliation_with_attention_pruning.data import (
    M5DataLoader,
    M5Preprocessor,
    HierarchyBuilder,
)
from adaptive_hierarchical_reconciliation_with_attention_pruning.models import (
    HierarchicalForecastModel,
)
from adaptive_hierarchical_reconciliation_with_attention_pruning.evaluation import (
    RMSSEMetric,
    HierarchicalCoherenceMetric,
    PruningRatioMetric,
    ForecastMetrics,
    ResultsAnalyzer,
)
from adaptive_hierarchical_reconciliation_with_attention_pruning.utils import (
    load_config,
    setup_logging,
    set_random_seeds,
    get_device,
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate hierarchical forecasting model"
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
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for results",
    )
    return parser.parse_args()


def load_test_data(config):
    """Load test data.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (test_loader, aggregation_matrix, preprocessor, sales_data)
    """
    logger.info("Loading test data...")

    # Load data
    data_loader = M5DataLoader(data_dir=config.get("data_dir", "data/m5"))
    sales_df, calendar_df, hierarchy_df = data_loader.load_or_generate(
        num_items=config.get("num_items", 100),
        num_days=config.get("num_days", 1000),
    )

    # Build hierarchy
    hierarchy_builder = HierarchyBuilder(hierarchy_df)
    aggregation_matrix = hierarchy_builder.build_aggregation_matrix()

    # Preprocess
    preprocessor = M5Preprocessor(
        lookback_window=config.get("lookback_window", 28),
        forecast_horizon=config.get("forecast_horizon", 7),
        detect_changepoints=config.get("detect_changepoints", True),
    )

    X, y, cal_features = preprocessor.prepare_sequences(sales_df, calendar_df)

    # Use last 20% as test set
    test_ratio = config.get("test_ratio", 0.2)
    num_samples = X.shape[0]
    test_start = int(num_samples * (1 - test_ratio))

    X_test = X[test_start:]
    y_test = y[test_start:]
    cal_test = cal_features[test_start:]

    # Get sales data for RMSSE computation
    sales_cols = [col for col in sales_df.columns if col.startswith("d_")]
    sales_data = sales_df[sales_cols].values

    # Create test loader
    test_dataset = TensorDataset(X_test, y_test, cal_test)
    test_loader = DataLoader(
        test_dataset, batch_size=config.get("batch_size", 32), shuffle=False
    )

    return (
        test_loader,
        torch.FloatTensor(aggregation_matrix),
        preprocessor,
        sales_data,
    )


def load_model(checkpoint_path, config, num_bottom_level, num_total_series, device):
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        config: Configuration dictionary
        num_bottom_level: Number of bottom-level series
        num_total_series: Total number of series
        device: Device to load model on

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")

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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info("Model loaded successfully")
    return model


def evaluate_model(model, test_loader, aggregation_matrix, device):
    """Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        aggregation_matrix: Aggregation matrix
        device: Device

    Returns:
        Tuple of (predictions, targets, pruning_masks)
    """
    logger.info("Evaluating model...")

    all_bottom_predictions = []
    all_reconciled_predictions = []
    all_targets = []
    all_pruning_masks = []

    with torch.no_grad():
        for x, y, cal_features in test_loader:
            x = x.to(device)
            y = y.to(device)
            cal_features = cal_features.to(device)

            outputs = model(
                x, cal_features, aggregation_matrix.to(device), hard_pruning=True
            )

            all_bottom_predictions.append(
                outputs["bottom_predictions"].cpu().numpy()
            )
            all_reconciled_predictions.append(
                outputs["reconciled_predictions"].cpu().numpy()
            )
            all_targets.append(y.cpu().numpy())

            if outputs["pruning_mask"] is not None:
                all_pruning_masks.append(outputs["pruning_mask"].cpu().numpy())

    # Concatenate results
    bottom_predictions = np.concatenate(all_bottom_predictions, axis=0)
    reconciled_predictions = np.concatenate(all_reconciled_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    pruning_mask = all_pruning_masks[0] if all_pruning_masks else None

    return bottom_predictions, reconciled_predictions, targets, pruning_mask


def compute_metrics(
    bottom_predictions, reconciled_predictions, targets, aggregation_matrix, pruning_mask, sales_data
):
    """Compute evaluation metrics.

    Args:
        bottom_predictions: Bottom-level predictions
        reconciled_predictions: Reconciled predictions
        targets: Ground truth
        aggregation_matrix: Aggregation matrix
        pruning_mask: Pruning mask
        sales_data: Historical sales data

    Returns:
        Dictionary of metrics
    """
    logger.info("Computing metrics...")

    # Initialize metrics
    rmsse_metric = RMSSEMetric(train_data=sales_data[:, :-100])  # Use training portion
    coherence_metric = HierarchicalCoherenceMetric(aggregation_matrix)
    pruning_metric = PruningRatioMetric()
    forecast_metrics = ForecastMetrics()

    # Compute RMSSE on bottom-level predictions
    rmsse = rmsse_metric.compute(
        bottom_predictions.reshape(-1, bottom_predictions.shape[-1]),
        targets.reshape(-1, targets.shape[-1]),
    )

    # Compute hierarchical coherence
    # Use mean predictions across batch
    bottom_mean = bottom_predictions.mean(axis=0)  # (horizon,) -> needs to be (num_items, horizon)
    reconciled_mean = reconciled_predictions.mean(axis=0)  # (num_total, horizon)

    # For coherence, we need proper shapes
    # Reshape bottom predictions for coherence computation
    num_bottom = aggregation_matrix.shape[0]
    if bottom_predictions.shape[0] >= num_bottom:
        bottom_for_coherence = bottom_predictions[:num_bottom].T  # (num_bottom, horizon)
        reconciled_for_coherence = reconciled_predictions[0].T  # (num_total, horizon)
    else:
        # Pad if needed
        bottom_for_coherence = np.zeros((num_bottom, bottom_predictions.shape[1]))
        bottom_for_coherence[: bottom_predictions.shape[0]] = bottom_predictions.T
        reconciled_for_coherence = reconciled_predictions[0].T

    coherence = coherence_metric.compute(bottom_for_coherence, reconciled_for_coherence)

    # Compute pruning ratio
    pruning_ratio = pruning_metric.compute(pruning_mask)

    # Other forecast metrics on bottom-level predictions
    mae = forecast_metrics.mae(bottom_predictions, targets)
    rmse = forecast_metrics.rmse(bottom_predictions, targets)
    mape = forecast_metrics.mape(bottom_predictions, targets)

    metrics = {
        "RMSSE": float(rmsse),
        "hierarchical_coherence": float(coherence),
        "pruning_ratio": float(pruning_ratio),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
    }

    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level="INFO", log_file="logs/evaluation.log")
    logger.info("Starting evaluation pipeline")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Set random seeds
        set_random_seeds(config.get("seed", 42))

        # Get device
        device = get_device()
        logger.info(f"Using device: {device}")

        # Load test data
        test_loader, aggregation_matrix, preprocessor, sales_data = load_test_data(
            config
        )

        num_bottom_level = config.get("num_items", 100)
        num_total_series = aggregation_matrix.shape[1]

        # Load model
        model = load_model(
            args.checkpoint, config, num_bottom_level, num_total_series, device
        )

        # Evaluate
        bottom_predictions, reconciled_predictions, targets, pruning_mask = (
            evaluate_model(model, test_loader, aggregation_matrix, device)
        )

        # Compute metrics
        metrics = compute_metrics(
            bottom_predictions,
            reconciled_predictions,
            targets,
            aggregation_matrix.numpy(),
            pruning_mask,
            sales_data,
        )

        # Print results
        logger.info("\n" + "=" * 60)
        logger.info("Evaluation Results")
        logger.info("=" * 60)
        for metric_name, value in sorted(metrics.items()):
            logger.info(f"{metric_name:.<40} {value:.4f}")
        logger.info("=" * 60)

        # Save results
        analyzer = ResultsAnalyzer(results_dir=args.output_dir)
        analyzer.save_metrics(metrics, filename="evaluation_metrics.json")
        analyzer.create_summary_report(metrics, filename="evaluation_summary.txt")
        analyzer.plot_predictions(
            bottom_predictions[:5], targets[:5], filename="predictions.png"
        )

        # Save predictions
        np.save(
            Path(args.output_dir) / "bottom_predictions.npy", bottom_predictions
        )
        np.save(
            Path(args.output_dir) / "reconciled_predictions.npy",
            reconciled_predictions,
        )

        logger.info(f"Results saved to {args.output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
