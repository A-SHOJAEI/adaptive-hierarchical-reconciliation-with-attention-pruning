# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

## Running the Project

### 1. Train the Model (Full Version with Attention & Pruning)

```bash
python scripts/train.py --config configs/default.yaml
```

Expected output:
- Training logs in console
- Best model saved to `checkpoints/best_model.pt`
- Training history saved to `results/training_history.csv`
- Training curves plot saved to `results/training_curves.png`

### 2. Train Baseline (Ablation - No Pruning)

```bash
python scripts/train.py --config configs/ablation.yaml
```

This trains a baseline model without structured pruning for comparison.

### 3. Evaluate the Model

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```

Expected output:
- Evaluation metrics printed to console
- Metrics saved to `results/evaluation_metrics.json`
- Summary report saved to `results/evaluation_summary.txt`
- Predictions plot saved to `results/predictions.png`

### 4. Make Predictions on New Data

```bash
# Create sample input data (28 timesteps)
python -c "import numpy as np; np.save('input.npy', np.random.randn(28, 1))"

# Run prediction
python scripts/predict.py --checkpoint checkpoints/best_model.pt --input input.npy

# Output saved to predictions.json
cat predictions.json
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
```

## Project Structure Overview

```
adaptive-hierarchical-reconciliation-with-attention-pruning/
├── configs/               # Configuration files
│   ├── default.yaml      # Full model with pruning
│   └── ablation.yaml     # Baseline without pruning
├── scripts/              # Executable scripts
│   ├── train.py         # Training pipeline
│   ├── evaluate.py      # Evaluation
│   └── predict.py       # Inference
├── src/                 # Source code
│   └── adaptive_hierarchical_reconciliation_with_attention_pruning/
│       ├── data/        # Data loading
│       ├── models/      # Model architecture
│       ├── training/    # Training utilities
│       ├── evaluation/  # Metrics and analysis
│       └── utils/       # Helpers
└── tests/               # Test suite
```

## Key Configuration Parameters

Edit `configs/default.yaml` to customize:

```yaml
# Model architecture
hidden_dim: 64              # Hidden layer size
num_layers: 2               # Number of LSTM layers
num_attention_heads: 4      # Multi-head attention heads

# Training
num_epochs: 50              # Maximum epochs
learning_rate: 0.001        # Initial learning rate
batch_size: 32              # Batch size

# Novel features
enable_pruning: true        # Enable structured pruning
target_pruning_ratio: 0.4   # Target 40% pruning
coherence_weight: 0.3       # Hierarchical coherence weight
```

## Expected Results

Target metrics on M5-style hierarchical sales data:

| Metric | Target | Description |
|--------|--------|-------------|
| RMSSE | 0.65 | Root Mean Scaled Squared Error |
| Hierarchical Coherence | 0.90 | Consistency across hierarchy levels |
| Pruning Ratio | 0.40 | Fraction of constraints pruned |

## Troubleshooting

**Issue**: Import errors
- **Solution**: Make sure you're running from the project root directory

**Issue**: CUDA out of memory
- **Solution**: Reduce `batch_size` in config file (try 16 or 8)

**Issue**: MLflow connection error
- **Solution**: This is normal - MLflow tracking is optional and wrapped in try/except

**Issue**: Tests fail
- **Solution**: Run `pip install -r requirements.txt` to ensure all dependencies are installed

## Next Steps

1. Modify configurations in `configs/default.yaml`
2. Experiment with different model architectures
3. Compare full model vs baseline ablation
4. Analyze results in `results/` directory
5. Extend with real M5 competition data (download from Kaggle)

## Citation

If you use this code, please cite:

```
Adaptive Hierarchical Reconciliation with Attention Pruning
Author: Alireza Shojaei
Year: 2026
License: MIT
```
