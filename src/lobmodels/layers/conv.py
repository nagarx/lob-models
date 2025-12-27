"""
Convolutional building blocks.

The ConvBlock is the fundamental building block for spatial feature
extraction in DeepLOB and similar architectures.

Architecture per DeepLOB paper (Zhang et al. 2019, Section IV.B):
    Conv2d → Activation → BatchNorm

Key insight from paper:
    "The first layer essentially summarizes information between price and 
    volume {p(i), v(i)} at each order book level."
    
    This is achieved with (1,2) kernels that span price-volume pairs.

Design Principles (RULE.md):
- Configuration-driven (§4)
- Modular and composable (§3)
- Each block independently testable (§5)
"""

import torch
import torch.nn as nn
from typing import Tuple, Union, List

from lobmodels.config import ConvBlockConfig, ActivationType
from lobmodels.layers.activations import get_activation


class ConvBlock(nn.Module):
    """
    Configurable convolutional block: Conv2d → Activation → BatchNorm.
    
    This is the basic building block used throughout DeepLOB's
    convolutional encoder.
    
    Args:
        config: ConvBlockConfig with layer parameters
    
    Input shape:  [batch, in_channels, height, width]
    Output shape: [batch, out_channels, H', W']
        where H', W' depend on kernel, stride, and padding
    
    Example:
        >>> config = ConvBlockConfig(
        ...     in_channels=1, out_channels=32,
        ...     kernel_size=(1, 2), stride=(1, 2),
        ...     activation=ActivationType.LEAKY_RELU
        ... )
        >>> block = ConvBlock(config)
        >>> x = torch.randn(32, 1, 100, 40)
        >>> y = block(x)
        >>> y.shape
        torch.Size([32, 32, 100, 20])
    """
    
    def __init__(self, config: ConvBlockConfig):
        super().__init__()
        self.config = config
        
        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding if isinstance(config.padding, str) else config.padding,
        )
        
        # Activation
        self.activation = get_activation(
            config.activation,
            leaky_relu_slope=config.leaky_relu_slope,
        )
        
        # Batch normalization (optional)
        self.batchnorm = (
            nn.BatchNorm2d(config.out_channels)
            if config.use_batchnorm
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Conv → Activation → BatchNorm.
        
        Args:
            x: Input tensor [batch, in_channels, height, width]
        
        Returns:
            Output tensor [batch, out_channels, H', W']
        """
        x = self.conv(x)
        x = self.activation(x)
        x = self.batchnorm(x)
        return x
    
    def extra_repr(self) -> str:
        return (
            f"in={self.config.in_channels}, out={self.config.out_channels}, "
            f"kernel={self.config.kernel_size}, stride={self.config.stride}, "
            f"act={self.config.activation.value}"
        )


class ConvStack(nn.Module):
    """
    Stack of ConvBlocks applied sequentially.
    
    Used to build the three convolutional stages in DeepLOB.
    
    Args:
        configs: List of ConvBlockConfig for each layer in the stack
    
    Example:
        >>> configs = [
        ...     ConvBlockConfig(1, 32, (1, 2), (1, 2)),  # Reduce width
        ...     ConvBlockConfig(32, 32, (4, 1)),         # Temporal conv
        ...     ConvBlockConfig(32, 32, (4, 1)),         # Temporal conv
        ... ]
        >>> stack = ConvStack(configs)
    """
    
    def __init__(self, configs: List[ConvBlockConfig]):
        super().__init__()
        
        if not configs:
            raise ValueError("configs list cannot be empty")
        
        self.blocks = nn.ModuleList([ConvBlock(cfg) for cfg in configs])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply each block sequentially."""
        for block in self.blocks:
            x = block(x)
        return x


def compute_output_size(
    input_size: Tuple[int, int],
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int] = (1, 1),
    padding: Union[str, Tuple[int, int]] = (0, 0),
) -> Tuple[int, int]:
    """
    Compute output spatial dimensions after convolution.
    
    Formula: floor((input + 2*padding - kernel) / stride) + 1
    
    Args:
        input_size: (height, width) of input
        kernel_size: (kernel_h, kernel_w)
        stride: (stride_h, stride_w)
        padding: 'same', 'valid', or (pad_h, pad_w)
    
    Returns:
        (output_height, output_width)
    
    Example:
        >>> compute_output_size((100, 40), (1, 2), (1, 2), (0, 0))
        (100, 20)
    """
    H_in, W_in = input_size
    kH, kW = kernel_size
    sH, sW = stride
    
    if padding == 'same':
        return (H_in, W_in)
    elif padding == 'valid':
        pH, pW = 0, 0
    else:
        pH, pW = padding
    
    H_out = (H_in + 2 * pH - kH) // sH + 1
    W_out = (W_in + 2 * pW - kW) // sW + 1
    
    return (H_out, W_out)

