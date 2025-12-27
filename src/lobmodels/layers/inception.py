"""
Inception Module for multi-scale temporal pattern extraction.

The Inception Module is a key component of DeepLOB that captures
patterns at multiple temporal scales simultaneously.

Reference:
    Zhang et al. (2019), Section IV.B.b:
    "We can capture dynamic behaviours over multiple timescales by using 
    Inception Modules to wrap several convolutions together."
    
    Original Inception: Szegedy et al. (2015) "Going deeper with convolutions"

Architecture (Figure 4 of DeepLOB paper):
    
    Input (32 channels)
         │
    ┌────┼────┬────────────┐
    │    │    │            │
    │  1×1  1×1         MaxPool(3×1)
    │    │    │            │
    │  3×1  5×1          1×1
    │    │    │            │
    └────┼────┴────────────┘
         │
      Concat (64+64+64 = 192 channels)

Design Principles (RULE.md):
- Configuration-driven branch sizes (§4)
- Each branch independently testable (§5)
- Clear documentation of tensor dimensions (§8)
"""

import torch
import torch.nn as nn
from typing import Tuple

from lobmodels.config import InceptionConfig, ActivationType
from lobmodels.layers.activations import get_activation


class InceptionBranch(nn.Module):
    """
    Single branch of the Inception Module.
    
    Structure: [Optional 1×1 reduce] → [Main conv or pool] → Activation → BatchNorm
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        activation: ActivationType = ActivationType.LEAKY_RELU,
        leaky_relu_slope: float = 0.01,
        use_batchnorm: bool = True,
        use_1x1_reduce: bool = True,
    ):
        super().__init__()
        
        layers = []
        
        # 1×1 convolution for channel reduction (Network-in-Network)
        if use_1x1_reduce:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding='same'),
                get_activation(activation, leaky_relu_slope),
            ])
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            conv_in_channels = out_channels
        else:
            conv_in_channels = in_channels
        
        # Main convolution (temporal pattern)
        layers.extend([
            nn.Conv2d(conv_in_channels, out_channels, kernel_size=kernel_size, padding='same'),
            get_activation(activation, leaky_relu_slope),
        ])
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        self.branch = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class InceptionPoolBranch(nn.Module):
    """
    Pooling branch of the Inception Module.
    
    Structure: MaxPool(3×1) → 1×1 conv → Activation → BatchNorm
    
    The pooling captures local maximum features, complementing
    the convolutional branches that capture local patterns.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_kernel: int = 3,
        activation: ActivationType = ActivationType.LEAKY_RELU,
        leaky_relu_slope: float = 0.01,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        
        # MaxPool with same padding (stride=1 to preserve temporal dimension)
        self.pool = nn.MaxPool2d(
            kernel_size=(pool_kernel, 1),
            stride=(1, 1),
            padding=(pool_kernel // 2, 0),  # Same padding for height
        )
        
        # 1×1 convolution after pooling
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding='same'),
            get_activation(activation, leaky_relu_slope),
        ]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.conv(x)
        return x


class InceptionModule(nn.Module):
    """
    Inception Module for multi-scale temporal feature extraction.
    
    Implements the Inception Module from DeepLOB (Figure 4):
    - Branch 1: 1×1 → 3×1 convolution (short-term patterns, ~3 timesteps)
    - Branch 2: 1×1 → 5×1 convolution (medium-term patterns, ~5 timesteps)
    - Branch 3: MaxPool(3×1) → 1×1 (local maximum features)
    
    All branches produce the same number of channels and are concatenated.
    
    Args:
        config: InceptionConfig with module parameters
    
    Input shape:  [batch, in_channels, seq_len, 1]
    Output shape: [batch, out_channels, seq_len, 1]
        where out_channels = branch_filters × 3
    
    Example:
        >>> config = InceptionConfig(in_channels=32, branch_filters=64)
        >>> inception = InceptionModule(config)
        >>> x = torch.randn(32, 32, 82, 1)  # After conv blocks
        >>> y = inception(x)
        >>> y.shape
        torch.Size([32, 192, 82, 1])  # 64 × 3 = 192 channels
    
    Note:
        The paper uses 'same' padding to preserve the temporal dimension,
        which is important for the subsequent LSTM layer.
    """
    
    def __init__(self, config: InceptionConfig):
        super().__init__()
        self.config = config
        
        # Branch 1: Short-term patterns (1×1 → 3×1)
        self.branch_short = InceptionBranch(
            in_channels=config.in_channels,
            out_channels=config.branch_filters,
            kernel_size=(config.short_kernel, 1),
            activation=config.activation,
            leaky_relu_slope=config.leaky_relu_slope,
            use_batchnorm=config.use_batchnorm,
            use_1x1_reduce=True,
        )
        
        # Branch 2: Medium-term patterns (1×1 → 5×1)
        self.branch_medium = InceptionBranch(
            in_channels=config.in_channels,
            out_channels=config.branch_filters,
            kernel_size=(config.medium_kernel, 1),
            activation=config.activation,
            leaky_relu_slope=config.leaky_relu_slope,
            use_batchnorm=config.use_batchnorm,
            use_1x1_reduce=True,
        )
        
        # Branch 3: Pooling branch (MaxPool → 1×1)
        self.branch_pool = InceptionPoolBranch(
            in_channels=config.in_channels,
            out_channels=config.branch_filters,
            pool_kernel=config.pool_kernel,
            activation=config.activation,
            leaky_relu_slope=config.leaky_relu_slope,
            use_batchnorm=config.use_batchnorm,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply all three branches and concatenate outputs.
        
        Args:
            x: Input tensor [batch, in_channels, seq_len, 1]
        
        Returns:
            Concatenated output [batch, branch_filters×3, seq_len, 1]
        """
        # Apply each branch
        out_short = self.branch_short(x)    # [B, branch_filters, T, 1]
        out_medium = self.branch_medium(x)  # [B, branch_filters, T, 1]
        out_pool = self.branch_pool(x)      # [B, branch_filters, T, 1]
        
        # Concatenate along channel dimension
        out = torch.cat([out_short, out_medium, out_pool], dim=1)
        
        return out
    
    def extra_repr(self) -> str:
        return (
            f"in={self.config.in_channels}, "
            f"out={self.config.out_channels}, "
            f"branches=[{self.config.short_kernel}×1, "
            f"{self.config.medium_kernel}×1, pool-{self.config.pool_kernel}×1]"
        )

