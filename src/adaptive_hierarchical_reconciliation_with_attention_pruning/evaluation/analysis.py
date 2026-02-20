"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzer for hierarchical forecasting results."""

    def __init__(self, results_dir: str = "results"):
        """Initialize results analyzer.

        Args:
            results_dir: Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ResultsAnalyzer: {results_dir}")

    def save_metrics(
        self, metrics: Dict[str, float], filename: str = "metrics.json"
    ) -> None:
        """Save metrics to JSON file.

        Args:
            metrics: Dictionary of metrics
            filename: Output filename
        """
        output_path = self.results_dir / filename

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {output_path}")

    def save_training_history(
        self, history: Dict[str, List[float]], filename: str = "training_history.csv"
    ) -> None:
        """Save training history to CSV.

        Args:
            history: Training history dictionary
            filename: Output filename
        """
        output_path = self.results_dir / filename

        df = pd.DataFrame(history)
        df.to_csv(output_path, index=False)

        logger.info(f"Saved training history to {output_path}")

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        filename: str = "training_curves.png",
    ) -> None:
        """Plot training curves.

        Args:
            history: Training history dictionary
            filename: Output filename
        """
        output_path = self.results_dir / filename

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Loss curves
        axes[0, 0].plot(history["train_loss"], label="Train Loss")
        axes[0, 0].plot(history["val_loss"], label="Val Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Total Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Forecast loss
        axes[0, 1].plot(
            history["train_forecast_loss"], label="Train Forecast Loss"
        )
        axes[0, 1].plot(history["val_forecast_loss"], label="Val Forecast Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Forecast Loss")
        axes[0, 1].set_title("Forecast Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Coherence loss
        axes[1, 0].plot(
            history["train_coherence_loss"], label="Train Coherence Loss"
        )
        axes[1, 0].plot(
            history["val_coherence_loss"], label="Val Coherence Loss"
        )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Coherence Loss")
        axes[1, 0].set_title("Coherence Loss")
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Learning rate
        axes[1, 1].plot(history["learning_rate"])
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved training curves to {output_path}")

    def plot_predictions(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        num_samples: int = 5,
        filename: str = "predictions.png",
    ) -> None:
        """Plot sample predictions vs targets.

        Args:
            predictions: Predictions (num_samples, horizon)
            targets: Targets (num_samples, horizon)
            num_samples: Number of samples to plot
            filename: Output filename
        """
        output_path = self.results_dir / filename

        num_samples = min(num_samples, predictions.shape[0])
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples))

        if num_samples == 1:
            axes = [axes]

        for i in range(num_samples):
            axes[i].plot(targets[i], label="Target", marker="o")
            axes[i].plot(predictions[i], label="Prediction", marker="x")
            axes[i].set_xlabel("Time Step")
            axes[i].set_ylabel("Value")
            axes[i].set_title(f"Sample {i+1}")
            axes[i].legend()
            axes[i].grid(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved predictions plot to {output_path}")

    def analyze_hierarchy_levels(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        level_indices: Dict[str, tuple],
        filename: str = "hierarchy_analysis.csv",
    ) -> pd.DataFrame:
        """Analyze performance at each hierarchy level.

        Args:
            predictions: Predictions at all levels (num_total, horizon)
            targets: Targets at bottom level (num_bottom, horizon)
            level_indices: Dictionary mapping level names to index ranges
            filename: Output filename

        Returns:
            DataFrame with per-level metrics
        """
        results = []

        for level_name, (start_idx, end_idx) in level_indices.items():
            if level_name == "item":
                # For bottom level, compare directly
                level_preds = predictions[start_idx:end_idx]
                level_targets = targets
            else:
                # For aggregated levels, we would need aggregated targets
                # Skip for now or compute from bottom level
                continue

            # Compute metrics
            mae = np.mean(np.abs(level_preds - level_targets))
            rmse = np.sqrt(np.mean((level_preds - level_targets) ** 2))

            results.append(
                {
                    "level": level_name,
                    "num_series": end_idx - start_idx,
                    "mae": mae,
                    "rmse": rmse,
                }
            )

        df = pd.DataFrame(results)
        output_path = self.results_dir / filename
        df.to_csv(output_path, index=False)

        logger.info(f"Saved hierarchy analysis to {output_path}")
        return df

    def create_summary_report(
        self, metrics: Dict[str, float], filename: str = "summary_report.txt"
    ) -> None:
        """Create a summary report.

        Args:
            metrics: Dictionary of metrics
            filename: Output filename
        """
        output_path = self.results_dir / filename

        with open(output_path, "w") as f:
            f.write("=" * 60 + "\n")
            f.write("Hierarchical Forecasting Results Summary\n")
            f.write("=" * 60 + "\n\n")

            f.write("Key Metrics:\n")
            f.write("-" * 60 + "\n")
            for metric_name, value in sorted(metrics.items()):
                f.write(f"{metric_name:.<40} {value:.4f}\n")

            f.write("\n" + "=" * 60 + "\n")

        logger.info(f"Saved summary report to {output_path}")
