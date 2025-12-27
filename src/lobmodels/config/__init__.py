"""
Configuration system for lob-models.

All model hyperparameters are defined via configuration dataclasses.
This enables:
- Type-safe parameter definition
- Serialization for experiment tracking
- Validation at construction time
- IDE autocomplete support

Usage:
    >>> from lobmodels.config import DeepLOBConfig
    >>> config = DeepLOBConfig(num_levels=10, lstm_hidden=64)
    >>> model = DeepLOB(config)
"""

from lobmodels.config.base import (
    # Base configs
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

__all__ = [
    # Base
    "BaseConfig",
    # Layer configs
    "ConvBlockConfig",
    "InceptionConfig",
    "TemporalEncoderConfig",
    # Model configs
    "DeepLOBConfig",
    # Enums
    "ActivationType",
    "FeatureLayout",
]

