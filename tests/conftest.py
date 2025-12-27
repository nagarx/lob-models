"""
Pytest configuration and shared fixtures.

Provides common fixtures for testing:
- Random seed management for reproducibility
- Sample data generators
- Device configuration
"""

import pytest
import torch
import numpy as np
import random


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility in all tests."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield


@pytest.fixture
def device():
    """Get available device (CPU for CI, GPU if available locally)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    """Default batch size for tests."""
    return 32


@pytest.fixture
def sequence_length():
    """Default sequence length for tests."""
    return 100


@pytest.fixture
def num_levels():
    """Default number of LOB levels."""
    return 10


@pytest.fixture
def num_features_benchmark():
    """Number of features in benchmark mode (40 = 10 levels × 4)."""
    return 40


@pytest.fixture
def num_features_extended():
    """Number of features in extended mode (98)."""
    return 98


@pytest.fixture
def sample_grouped_input(batch_size, sequence_length, num_levels):
    """
    Generate sample input in GROUPED layout.
    
    Shape: [batch, seq_len, 4*num_levels]
    Layout: [bid_prices(L), ask_prices(L), bid_sizes(L), ask_sizes(L)]
    """
    return torch.randn(batch_size, sequence_length, 4 * num_levels)


@pytest.fixture
def sample_fi2010_input(batch_size, sequence_length, num_levels):
    """
    Generate sample input in FI2010 layout.
    
    Shape: [batch, seq_len, 4*num_levels]
    Layout: [p_a0, v_a0, p_b0, v_b0, ..., p_a9, v_a9, p_b9, v_b9]
    """
    return torch.randn(batch_size, sequence_length, 4 * num_levels)

