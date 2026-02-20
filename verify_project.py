#!/usr/bin/env python3
"""Verify project completeness."""

import os
from pathlib import Path

# Required files checklist
required_files = {
    # Source code
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/__init__.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/data/__init__.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/data/loader.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/data/preprocessing.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/models/__init__.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/models/model.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/models/components.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/training/__init__.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/training/trainer.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/evaluation/__init__.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/evaluation/metrics.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/evaluation/analysis.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/utils/__init__.py": True,
    "src/adaptive_hierarchical_reconciliation_with_attention_pruning/utils/config.py": True,
    
    # Scripts
    "scripts/train.py": True,
    "scripts/evaluate.py": True,
    "scripts/predict.py": True,
    
    # Tests
    "tests/__init__.py": True,
    "tests/conftest.py": True,
    "tests/test_data.py": True,
    "tests/test_model.py": True,
    "tests/test_training.py": True,
    
    # Configs
    "configs/default.yaml": True,
    "configs/ablation.yaml": True,
    
    # Documentation
    "README.md": True,
    "LICENSE": True,
    "requirements.txt": True,
    "pyproject.toml": True,
    ".gitignore": True,
}

print("=" * 80)
print("PROJECT COMPLETENESS VERIFICATION")
print("=" * 80)

missing_files = []
existing_files = []

for file_path, required in required_files.items():
    exists = Path(file_path).exists()
    status = "✓" if exists else "✗"
    
    if exists:
        existing_files.append(file_path)
    else:
        missing_files.append(file_path)
    
    print(f"{status} {file_path}")

print("\n" + "=" * 80)
print(f"SUMMARY: {len(existing_files)}/{len(required_files)} files present")
print("=" * 80)

if missing_files:
    print("\nMISSING FILES:")
    for f in missing_files:
        print(f"  - {f}")
    exit(1)
else:
    print("\n✓ ALL REQUIRED FILES PRESENT!")
    print("✓ Project is complete and ready for use!")
    exit(0)
