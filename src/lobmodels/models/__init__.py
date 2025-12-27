"""
Model architectures for LOB price prediction.

Available models:
- DeepLOB: CNN-LSTM hybrid (Zhang et al. 2019)

All models inherit from BaseModel and provide a consistent interface.

Usage:
    >>> from lobmodels import DeepLOB, DeepLOBConfig
    >>> config = DeepLOBConfig(mode="benchmark")
    >>> model = DeepLOB(config)
    >>> 
    >>> # Or use factory function
    >>> from lobmodels import create_model
    >>> model = create_model("deeplob", config)
"""

from lobmodels.models.base import BaseModel
from lobmodels.models.deeplob import DeepLOB, create_deeplob

__all__ = [
    "BaseModel",
    "DeepLOB",
    "create_deeplob",
]

