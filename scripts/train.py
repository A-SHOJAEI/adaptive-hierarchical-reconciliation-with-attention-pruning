#!/usr/bin/env python
"""Training script for hierarchical forecasting model."""

import argparse
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
    HierarchicalCoherenceLoss,
)
from adaptive_hierarchical_reconciliation_with_attention_pruning.training import (
    HierarchicalTrainer,
)
from adaptive_hierarchical_reconciliation_with_attention_pruning.evaluation import (
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
        description="Train hierarchical forecasting model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


def prepare_data(config):
    """Prepare training and validation data.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, aggregation_matrix, preprocessor)
    """
    logger.info("Loading and preprocessing data...")

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

    logger.info(f"Prepared data: X={X.shape}, y={y.shape}")

    # Split into train/val
    train_ratio = config.get("train_ratio", 0.8)
    num_samples = X.shape[0]
    train_size = int(num_samples * train_ratio)

    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    cal_train, cal_val = cal_features[:train_size], cal_features[train_size:]

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train, cal_train)
    val_dataset = TensorDataset(X_val, y_val, cal_val)

    batch_size = config.get("batch_size", 32)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        torch.FloatTensor(aggregation_matrix),
        preprocessor,
    )


def create_model(config, num_bottom_level, num_total_series):
    """Create the hierarchical forecasting model.

    Args:
        config: Configuration dictionary
        num_bottom_level: Number of bottom-level series
        num_total_series: Total number of series

    Returns:
        Model instance
    """
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

    logger.info(
        f"Created model with {sum(p.numel() for p in model.parameters())} parameters"
    )

    return model


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_level=args.log_level, log_file="logs/training.log")
    logger.info("Starting training pipeline")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Set random seeds
        seed = config.get("seed", 42)
        set_random_seeds(seed)
        logger.info(f"Set random seed to {seed}")

        # Get device
        device = get_device()
        logger.info(f"Using device: {device}")

        # Prepare data
        train_loader, val_loader, aggregation_matrix, preprocessor = prepare_data(
            config
        )

        num_bottom_level = config.get("num_items", 100)
        num_total_series = aggregation_matrix.shape[1]

        # Create model
        model = create_model(config, num_bottom_level, num_total_series)

        # Create loss function
        loss_fn = HierarchicalCoherenceLoss(
            coherence_weight=config.get("coherence_weight", 0.3),
            forecast_weight=config.get("forecast_weight", 0.7),
            use_rmsse=config.get("use_rmsse", True),
        )

        # Create trainer
        trainer = HierarchicalTrainer(
            model=model,
            loss_fn=loss_fn,
            aggregation_matrix=aggregation_matrix,
            learning_rate=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 0.0001),
            device=device,
            gradient_clip=config.get("gradient_clip", 1.0),
            patience=config.get("patience", 10),
            min_delta=config.get("min_delta", 0.0001),
        )

        # Initialize MLflow (wrapped in try/except as server may not be available)
        try:
            import mlflow

            mlflow.set_experiment(
                config.get("experiment_name", "hierarchical_forecasting")
            )
            mlflow.start_run()
            mlflow.log_params(config)
            logger.info("MLflow tracking initialized")
            use_mlflow = True
        except Exception as e:
            logger.warning(f"MLflow not available: {e}")
            use_mlflow = False

        # Train model
        num_epochs = config.get("num_epochs", 50)
        checkpoint_dir = config.get("checkpoint_dir", "checkpoints")

        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            checkpoint_dir=checkpoint_dir,
        )

        # Log final metrics
        final_metrics = {
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "best_val_loss": min(history["val_loss"]),
            "pruning_ratio": model.get_pruning_ratio(),
        }

        logger.info("Training completed!")
        logger.info(f"Final metrics: {final_metrics}")

        if use_mlflow:
            try:
                mlflow.log_metrics(final_metrics)
                mlflow.end_run()
            except Exception as e:
                logger.warning(f"Failed to log to MLflow: {e}")

        # Save results
        analyzer = ResultsAnalyzer(results_dir=config.get("results_dir", "results"))
        analyzer.save_training_history(history)
        analyzer.plot_training_curves(history)
        analyzer.save_metrics(final_metrics, filename="final_metrics.json")

        logger.info("Results saved successfully")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
