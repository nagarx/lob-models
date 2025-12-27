"""
Activation function utilities.

Provides a unified interface for creating activation functions
based on configuration strings/enums.

Design Principles (RULE.md):
- Configuration-driven activation selection (§4)
- Explicit parameters, no hidden defaults (§4)
"""

import torch.nn as nn
from typing import Optional

from lobmodels.config import ActivationType


def get_activation(
    activation: ActivationType | str,
    leaky_relu_slope: float = 0.01,
) -> nn.Module:
    """
    Get activation module from type specification.
    
    Args:
        activation: Activation type (enum or string)
        leaky_relu_slope: Negative slope for LeakyReLU
    
    Returns:
        PyTorch activation module
    
    Raises:
        ValueError: If activation type is not recognized
    
    Example:
        >>> act = get_activation(ActivationType.LEAKY_RELU, leaky_relu_slope=0.01)
        >>> x = torch.randn(32, 64)
        >>> y = act(x)
    """
    # Convert string to enum if needed
    if isinstance(activation, str):
        try:
            activation = ActivationType(activation)
        except ValueError:
            raise ValueError(
                f"Unknown activation type: '{activation}'. "
                f"Valid types: {[a.value for a in ActivationType]}"
            )
    
    if activation == ActivationType.RELU:
        return nn.ReLU(inplace=True)
    
    elif activation == ActivationType.LEAKY_RELU:
        return nn.LeakyReLU(negative_slope=leaky_relu_slope, inplace=True)
    
    elif activation == ActivationType.TANH:
        return nn.Tanh()
    
    elif activation == ActivationType.GELU:
        return nn.GELU()
    
    elif activation == ActivationType.NONE:
        return nn.Identity()
    
    else:
        raise ValueError(f"Unknown activation type: {activation}")

