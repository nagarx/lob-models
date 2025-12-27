"""
Feature layout transformation utilities.

Converts between different LOB feature layouts to enable:
1. Benchmark mode: Use exact FI-2010 layout for paper comparison
2. Extended mode: Use our rich 98-feature format

Layout Definitions:

    GROUPED (Our Rust pipeline output):
        [bid_prices(L), ask_prices(L), bid_sizes(L), ask_sizes(L)]
        Indices: bid_p[0:L], ask_p[L:2L], bid_s[2L:3L], ask_s[3L:4L]
        Total: 4L features (40 for L=10)
    
    FI2010 (Original DeepLOB paper):
        For each level: [price_ask, volume_ask, price_bid, volume_bid]
        [p_a0, v_a0, p_b0, v_b0, p_a1, v_a1, p_b1, v_b1, ..., p_a9, v_a9, p_b9, v_b9]
        Total: 4L features (40 for L=10)

The FI2010 layout is designed so that a (1,2) kernel with stride (1,2)
pairs adjacent price/volume features at each level.

Design Principles (RULE.md):
- Explicit transformations with clear documentation (§1, §8)
- Pure functions for testability (§5)
- No hidden state or side effects (§6)
"""

import torch
import torch.nn as nn
from typing import Literal

from lobmodels.config import FeatureLayout


def rearrange_grouped_to_fi2010(
    x: torch.Tensor,
    num_levels: int = 10,
) -> torch.Tensor:
    """
    Rearrange features from GROUPED layout to FI2010 layout.
    
    GROUPED:  [bid_p(L), ask_p(L), bid_s(L), ask_s(L)]
    FI2010:   [p_a0, v_a0, p_b0, v_b0, ..., p_aL-1, v_aL-1, p_bL-1, v_bL-1]
    
    Args:
        x: Input tensor of shape [..., 4*num_levels]
           Last dimension contains grouped features
        num_levels: Number of LOB levels (default: 10)
    
    Returns:
        Rearranged tensor of shape [..., 4*num_levels]
        Last dimension contains FI2010-layout features
    
    Example:
        >>> x = torch.randn(32, 100, 40)  # [batch, seq, features]
        >>> y = rearrange_grouped_to_fi2010(x, num_levels=10)
        >>> y.shape
        torch.Size([32, 100, 40])
    
    Note:
        This is a pure permutation - no values are modified, only reordered.
    """
    L = num_levels
    expected_features = 4 * L
    
    if x.shape[-1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features for {L} levels, "
            f"got {x.shape[-1]}"
        )
    
    # Extract grouped components
    # GROUPED layout: [bid_p(0:L), ask_p(L:2L), bid_s(2L:3L), ask_s(3L:4L)]
    bid_prices = x[..., 0:L]        # [*, L]
    ask_prices = x[..., L:2*L]      # [*, L]
    bid_sizes = x[..., 2*L:3*L]     # [*, L]
    ask_sizes = x[..., 3*L:4*L]     # [*, L]
    
    # Build FI2010 layout: for each level, [p_ask, v_ask, p_bid, v_bid]
    # Stack per level then interleave
    levels = []
    for i in range(L):
        levels.append(ask_prices[..., i:i+1])  # p_ask
        levels.append(ask_sizes[..., i:i+1])   # v_ask
        levels.append(bid_prices[..., i:i+1])  # p_bid
        levels.append(bid_sizes[..., i:i+1])   # v_bid
    
    # Concatenate along feature dimension
    return torch.cat(levels, dim=-1)


def rearrange_fi2010_to_grouped(
    x: torch.Tensor,
    num_levels: int = 10,
) -> torch.Tensor:
    """
    Rearrange features from FI2010 layout to GROUPED layout.
    
    This is the inverse of rearrange_grouped_to_fi2010.
    
    FI2010:   [p_a0, v_a0, p_b0, v_b0, ..., p_aL-1, v_aL-1, p_bL-1, v_bL-1]
    GROUPED:  [bid_p(L), ask_p(L), bid_s(L), ask_s(L)]
    
    Args:
        x: Input tensor of shape [..., 4*num_levels]
        num_levels: Number of LOB levels (default: 10)
    
    Returns:
        Rearranged tensor of shape [..., 4*num_levels]
    """
    L = num_levels
    expected_features = 4 * L
    
    if x.shape[-1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features for {L} levels, "
            f"got {x.shape[-1]}"
        )
    
    # Extract from FI2010 layout
    # FI2010: [p_a0, v_a0, p_b0, v_b0, p_a1, v_a1, p_b1, v_b1, ...]
    # Each level has 4 features at indices [4i, 4i+1, 4i+2, 4i+3]
    ask_prices = []
    ask_sizes = []
    bid_prices = []
    bid_sizes = []
    
    for i in range(L):
        base = 4 * i
        ask_prices.append(x[..., base:base+1])      # p_ask at 4i
        ask_sizes.append(x[..., base+1:base+2])     # v_ask at 4i+1
        bid_prices.append(x[..., base+2:base+3])    # p_bid at 4i+2
        bid_sizes.append(x[..., base+3:base+4])     # v_bid at 4i+3
    
    # Concatenate into GROUPED layout
    return torch.cat(
        [
            torch.cat(bid_prices, dim=-1),  # [*, L]
            torch.cat(ask_prices, dim=-1),  # [*, L]
            torch.cat(bid_sizes, dim=-1),   # [*, L]
            torch.cat(ask_sizes, dim=-1),   # [*, L]
        ],
        dim=-1
    )


class FeatureRearrangement(nn.Module):
    """
    PyTorch module for feature layout transformation.
    
    Use this as the first layer in a model to convert input features
    from one layout to another.
    
    Args:
        source: Source feature layout
        target: Target feature layout
        num_levels: Number of LOB levels
    
    Example:
        >>> rearrange = FeatureRearrangement(
        ...     source=FeatureLayout.GROUPED,
        ...     target=FeatureLayout.FI2010,
        ...     num_levels=10
        ... )
        >>> x = torch.randn(32, 100, 40)
        >>> y = rearrange(x)
        >>> y.shape
        torch.Size([32, 100, 40])
    
    Note:
        - If source == target, this is an identity operation
        - EXTENDED layout cannot be converted to FI2010/GROUPED (only first 40 features would be used)
    """
    
    def __init__(
        self,
        source: FeatureLayout,
        target: FeatureLayout,
        num_levels: int = 10,
    ):
        super().__init__()
        self.source = source
        self.target = target
        self.num_levels = num_levels
        
        # Validate conversion is possible
        if source == FeatureLayout.EXTENDED and target != FeatureLayout.EXTENDED:
            raise ValueError(
                f"Cannot convert from EXTENDED to {target}. "
                "EXTENDED layout has 98 features; FI2010/GROUPED have 40."
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feature layout transformation.
        
        Args:
            x: Input tensor [..., num_features]
        
        Returns:
            Transformed tensor [..., num_features]
        """
        # Identity case
        if self.source == self.target:
            return x
        
        # GROUPED -> FI2010
        if self.source == FeatureLayout.GROUPED and self.target == FeatureLayout.FI2010:
            return rearrange_grouped_to_fi2010(x, self.num_levels)
        
        # FI2010 -> GROUPED
        if self.source == FeatureLayout.FI2010 and self.target == FeatureLayout.GROUPED:
            return rearrange_fi2010_to_grouped(x, self.num_levels)
        
        # EXTENDED -> EXTENDED is identity (handled above)
        # Other conversions not supported
        raise NotImplementedError(
            f"Conversion from {self.source} to {self.target} not implemented"
        )
    
    def extra_repr(self) -> str:
        return f"source={self.source.value}, target={self.target.value}, num_levels={self.num_levels}"


def add_channel_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Add channel dimension for 2D convolution.
    
    Converts [batch, seq_len, features] to [batch, 1, seq_len, features]
    
    Args:
        x: Input tensor of shape [batch, seq_len, features]
    
    Returns:
        Tensor of shape [batch, 1, seq_len, features]
    """
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor [batch, seq, feat], got shape {x.shape}")
    return x.unsqueeze(1)


def remove_channel_dim(x: torch.Tensor) -> torch.Tensor:
    """
    Remove channel dimension after 2D convolution.
    
    Converts [batch, channels, seq_len, 1] to [batch, seq_len, channels]
    
    Args:
        x: Input tensor of shape [batch, channels, seq_len, 1]
    
    Returns:
        Tensor of shape [batch, seq_len, channels]
    """
    if x.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got shape {x.shape}")
    if x.shape[-1] != 1:
        raise ValueError(f"Expected last dim to be 1, got {x.shape[-1]}")
    # [batch, channels, seq_len, 1] -> [batch, seq_len, channels]
    return x.squeeze(-1).permute(0, 2, 1)

