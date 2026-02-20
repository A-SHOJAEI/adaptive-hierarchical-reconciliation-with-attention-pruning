# Adaptive Hierarchical Reconciliation with Attention Pruning

Hierarchical sales forecasting system combining bottom-up deep learning predictions with trainable attention-weighted reconciliation. Novel contribution: cross-level attention mechanism that dynamically learns optimal aggregation weights across hierarchy levels, with structured pruning to identify which hierarchical constraints matter most.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
python scripts/train.py --config configs/default.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

### Prediction

```bash
# Create sample input
python -c "import numpy as np; np.save('input.npy', np.random.randn(28, 1))"

# Run prediction
python scripts/predict.py --checkpoint checkpoints/best_model.pt --input input.npy
```

## Key Features

### Novel Contributions

1. **Attention-based Reconciliation**: Replaces traditional bottom-up summation with trainable cross-level attention that learns optimal aggregation weights
2. **Structured Pruning**: Identifies which hierarchical constraints are most important, achieving target pruning ratio of 40% while maintaining forecast accuracy
3. **Adaptive Forecast Windows**: Automatic structural break detection using change point detection for dynamic forecast window selection

### Architecture

- Bottom-level forecasting: Temporal CNN + LSTM for sequential feature extraction
- Attention reconciliation layer: Multi-head cross-level attention for hierarchical aggregation
- Structured pruning: Learnable importance scores for constraint selection
- Custom loss: Combines forecast accuracy (RMSSE) with hierarchical coherence

## Model Components

### HierarchicalForecastModel

Main model implementing:
- Temporal convolutional blocks with dilated convolutions
- LSTM for sequential dependencies
- Attention-weighted reconciliation across hierarchy levels
- Optional structured pruning of aggregation matrix

### Custom Components

- `AttentionReconciliationLayer`: Cross-level attention for aggregation
- `StructuredPruning`: Learnable masks for constraint selection
- `HierarchicalCoherenceLoss`: Multi-objective loss combining forecast accuracy and coherence

## Configuration

Two configurations provided:

- `configs/default.yaml`: Full model with attention reconciliation and pruning
- `configs/ablation.yaml`: Baseline without pruning for comparison

### Key Hyperparameters

```yaml
# Model
hidden_dim: 64
num_layers: 2
num_attention_heads: 4
enable_pruning: true
target_pruning_ratio: 0.4

# Training
learning_rate: 0.001
batch_size: 32
num_epochs: 50
gradient_clip: 1.0
patience: 10

# Loss
coherence_weight: 0.3
forecast_weight: 0.7
```

## Ablation Study

Compare full model vs baseline:

```bash
# Train full model
python scripts/train.py --config configs/default.yaml

# Train baseline (no pruning, reduced coherence weight)
python scripts/train.py --config configs/ablation.yaml

# Evaluate both
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --output-dir results_full
python scripts/evaluate.py --checkpoint checkpoints_ablation/best_model.pt --output-dir results_ablation
```

## Results

Run `python scripts/train.py` to reproduce. Expected performance on M5-style hierarchical sales data:

| Metric | Target | Description |
|--------|--------|-------------|
| RMSSE | 0.65 | Root Mean Scaled Squared Error |
| Hierarchical Coherence | 0.90 | Coherence between hierarchy levels (1.0 = perfect) |
| Pruning Ratio | 0.40 | Fraction of hierarchical constraints pruned |

## Project Structure

```
adaptive-hierarchical-reconciliation-with-attention-pruning/
├── src/
│   └── adaptive_hierarchical_reconciliation_with_attention_pruning/
│       ├── data/              # Data loading and preprocessing
│       ├── models/            # Model architecture and custom components
│       ├── training/          # Training loop with early stopping
│       ├── evaluation/        # Metrics and analysis
│       └── utils/             # Configuration and utilities
├── scripts/
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation with multiple metrics
│   └── predict.py            # Inference on new data
├── configs/
│   ├── default.yaml          # Full model configuration
│   └── ablation.yaml         # Baseline configuration
├── tests/                    # Comprehensive test suite
└── requirements.txt
```

## Testing

```bash
pytest tests/ --cov=src --cov-report=html
```

## Technical Details

### Hierarchy Structure

The model supports multi-level hierarchies:
- Total (aggregate of all)
- State level
- Store level
- Category level
- Item level (bottom)

### Training Features

- Cosine annealing learning rate schedule
- Early stopping with configurable patience
- Gradient clipping for stability
- MLflow tracking (optional, wrapped in try/except)
- Mixed precision support via PyTorch AMP
- Checkpoint saving for best model

### Evaluation Metrics

- **RMSSE**: Root Mean Scaled Squared Error (M5 competition metric)
- **Hierarchical Coherence**: Measures consistency across hierarchy levels
- **Pruning Ratio**: Fraction of constraints removed
- **MAE, RMSE, MAPE**: Standard forecast accuracy metrics

## License

MIT License - Copyright (c) 2026 Alireza Shojaei. See [LICENSE](LICENSE) for details.
