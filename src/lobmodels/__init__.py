"""
LOB Models: Deep learning architectures for Limit Order Book prediction.

A modular, configurable library for building and training neural networks
on Limit Order Book (LOB) data for price movement prediction.

Main Components:
    - DeepLOB: CNN-LSTM hybrid (Zhang et al. 2019)
    - Configurable building blocks (ConvBlock, InceptionModule, etc.)
    - Feature layout utilities for data transformation

Quick Start:
    >>> from lobmodels import DeepLOB, DeepLOBConfig
    >>> 
    >>> # Create model with default (benchmark) configuration
    >>> config = DeepLOBConfig(mode="benchmark")
    >>> model = DeepLOB(config)
    >>> 
    >>> # Forward pass
    >>> import torch
    >>> x = torch.randn(32, 100, 40)  # [batch, seq_len, features]
    >>> logits = model(x)
    >>> logits.shape
    torch.Size([32, 3])

Design Principles:
    - Configuration-driven: All hyperparameters via dataclass configs
    - Modular: Independent, composable building blocks
    - Testable: Each component has comprehensive tests
    - Documented: Clear docstrings with tensor shapes

Repository Structure:
    lobmodels/
    ├── config/          # Configuration dataclasses
    ├── layers/          # Building blocks (ConvBlock, Inception, LSTM)
    ├── models/          # Full architectures (DeepLOB)
    └── utils/           # Feature layout utilities

References:
    [1] Zhang, Zohren & Roberts (2019). "DeepLOB: Deep Convolutional Neural 
        Networks for Limit Order Books." IEEE Trans. Signal Processing.
"""

from lobmodels.version import __version__, get_version

# Configuration
from lobmodels.config import (
    # Base
    BaseConfig,
    # Layer configs
    ConvBlockConfig,
    InceptionConfig,
    TemporalEncoderConfig,
    # Model configs
    DeepLOBConfig,
    # Enums
    ActivationType,
    FeatureLayout,
)

# Layers
from lobmodels.layers import (
    get_activation,
    ConvBlock,
    InceptionModule,
    TemporalEncoder,
)

# Models
from lobmodels.models import (
    BaseModel,
    DeepLOB,
    create_deeplob,
)

# Utils
from lobmodels.utils import (
    rearrange_grouped_to_fi2010,
    rearrange_fi2010_to_grouped,
    FeatureRearrangement,
)


__all__ = [
    # Version
    "__version__",
    "get_version",
    # Configuration
    "BaseConfig",
    "ConvBlockConfig",
    "InceptionConfig",
    "TemporalEncoderConfig",
    "DeepLOBConfig",
    "ActivationType",
    "FeatureLayout",
    # Layers
    "get_activation",
    "ConvBlock",
    "InceptionModule",
    "TemporalEncoder",
    # Models
    "BaseModel",
    "DeepLOB",
    "create_deeplob",
    # Utils
    "rearrange_grouped_to_fi2010",
    "rearrange_fi2010_to_grouped",
    "FeatureRearrangement",
]

