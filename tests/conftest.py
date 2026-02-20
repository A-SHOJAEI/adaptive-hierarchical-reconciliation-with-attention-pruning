"""Pytest configuration and fixtures."""

import numpy as np
import pandas as pd
import pytest
import torch


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "seed": 42,
        "num_items": 10,
        "num_days": 100,
        "lookback_window": 7,
        "forecast_horizon": 3,
        "hidden_dim": 16,
        "num_layers": 1,
        "num_attention_heads": 2,
        "dropout": 0.1,
        "batch_size": 4,
        "learning_rate": 0.001,
        "enable_pruning": True,
        "target_pruning_ratio": 0.4,
    }


@pytest.fixture
def sample_sales_data():
    """Sample sales data for testing."""
    np.random.seed(42)
    num_items = 10
    num_days = 100

    data = []
    for item_id in range(num_items):
        row = {
            "item_id": f"item_{item_id}",
            "state_id": f"state_{item_id // 5}",
            "store_id": f"store_{item_id // 3}",
            "cat_id": f"cat_{item_id // 2}",
        }
        for day in range(num_days):
            row[f"d_{day+1}"] = np.random.gamma(2, 2)

        data.append(row)

    return pd.DataFrame(data)


@pytest.fixture
def sample_calendar_data():
    """Sample calendar data for testing."""
    num_days = 100
    dates = pd.date_range(start="2020-01-01", periods=num_days, freq="D")

    return pd.DataFrame(
        {
            "d": [f"d_{i+1}" for i in range(num_days)],
            "date": dates,
            "weekday": dates.weekday,
            "month": dates.month,
            "year": dates.year,
            "is_weekend": (dates.weekday >= 5).astype(int),
        }
    )


@pytest.fixture
def sample_hierarchy_data():
    """Sample hierarchy data for testing."""
    num_items = 10
    hierarchy_data = []

    for item_id in range(num_items):
        hierarchy_data.append(
            {
                "item_id": f"item_{item_id}",
                "state_id": f"state_{item_id // 5}",
                "store_id": f"store_{item_id // 3}",
                "cat_id": f"cat_{item_id // 2}",
                "level": "item",
            }
        )

    return pd.DataFrame(hierarchy_data)


@pytest.fixture
def sample_aggregation_matrix():
    """Sample aggregation matrix for testing."""
    num_items = 10
    num_total = 20  # Total across all hierarchy levels
    S = np.random.rand(num_items, num_total)
    return S


@pytest.fixture
def sample_sequences():
    """Sample input sequences for testing."""
    batch_size = 4
    lookback = 7
    horizon = 3

    X = torch.randn(batch_size, lookback, 1)
    y = torch.randn(batch_size, horizon)
    cal_features = torch.randn(batch_size, lookback, 3)

    return X, y, cal_features


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")
