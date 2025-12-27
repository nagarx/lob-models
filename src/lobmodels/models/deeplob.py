"""
DeepLOB: Deep Convolutional Neural Networks for Limit Order Books.

Reference:
    Zhang, Zohren & Roberts (2019).
    "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books."
    IEEE Transactions on Signal Processing, 67(11), 3001-3012.
    https://arxiv.org/abs/1808.03668

Architecture Overview:
    
    Input [B, T, F]
        │
        ▼
    Feature Rearrangement (GROUPED → FI2010 layout)
        │
        ▼
    Add Channel Dim → [B, 1, T, F]
        │
        ▼
    ┌───────────────────────────────┐
    │     Convolutional Encoder     │
    │  ┌─────────────────────────┐  │
    │  │ Conv Block 1            │  │
    │  │ • Conv(1×2, s=1×2)      │  │ ← Pairs price/volume
    │  │ • Conv(4×1) × 2         │  │ ← Temporal convs
    │  └─────────────────────────┘  │
    │  ┌─────────────────────────┐  │
    │  │ Conv Block 2            │  │
    │  │ • Conv(1×2, s=1×2)      │  │ ← Further reduction
    │  │ • Conv(4×1) × 2         │  │
    │  └─────────────────────────┘  │
    │  ┌─────────────────────────┐  │
    │  │ Conv Block 3            │  │
    │  │ • Conv(1×10)            │  │ ← Consolidate levels
    │  │ • Conv(4×1) × 2         │  │
    │  └─────────────────────────┘  │
    └───────────────────────────────┘
        │
        ▼ [B, 32, T', 1]
    ┌───────────────────────────────┐
    │      Inception Module         │
    │  • Branch 1: 1×1 → 3×1        │
    │  • Branch 2: 1×1 → 5×1        │
    │  • Branch 3: Pool → 1×1       │
    │  → Concat: 192 channels       │
    └───────────────────────────────┘
        │
        ▼ [B, 192, T', 1]
    Reshape → [B, T', 192]
        │
        ▼
    ┌───────────────────────────────┐
    │      LSTM Encoder             │
    │  • 64 hidden units            │
    │  • Last hidden state          │
    └───────────────────────────────┘
        │
        ▼ [B, 64]
    Linear → [B, num_classes]

Design Principles (RULE.md):
- Modular components (§3)
- Configuration-driven (§4)
- Explicit tensor shapes in comments (§8)
- Deterministic given same seed (§6)
"""

import torch
import torch.nn as nn
from typing import Optional

from lobmodels.config import (
    DeepLOBConfig,
    ConvBlockConfig,
    InceptionConfig,
    TemporalEncoderConfig,
    ActivationType,
    FeatureLayout,
)
from lobmodels.models.base import BaseModel
from lobmodels.layers import ConvBlock, InceptionModule, TemporalEncoder
from lobmodels.utils.feature_layout import (
    FeatureRearrangement,
    add_channel_dim,
    remove_channel_dim,
)


class ConvolutionalEncoder(nn.Module):
    """
    Convolutional encoder for spatial feature extraction.
    
    Implements the three conv blocks from DeepLOB paper:
    - Block 1: Pairs price/volume, temporal convolutions
    - Block 2: Further spatial reduction, temporal convolutions
    - Block 3: Consolidates all levels, temporal convolutions
    
    Args:
        num_levels: Number of LOB levels (10 for FI-2010)
        filters: Number of filters per conv layer
        leaky_relu_slope: Negative slope for LeakyReLU
    
    Input shape:  [batch, 1, seq_len, 4*num_levels]
    Output shape: [batch, filters, seq_len', 1]
        where seq_len' = seq_len - 18 (for default kernel sizes)
    """
    
    def __init__(
        self,
        num_levels: int = 10,
        filters: int = 32,
        leaky_relu_slope: float = 0.01,
    ):
        super().__init__()
        
        self.num_levels = num_levels
        self.filters = filters
        
        # =====================================================================
        # Block 1: LOB Structure Extraction
        # Paper: "summarizes information between price and volume {p(i), v(i)}"
        # =====================================================================
        # Conv(1×2, s=1×2): 40 → 20 features (pairs price/volume)
        # Conv(4×1): Temporal convolution
        # Conv(4×1): Temporal convolution
        self.block1 = nn.Sequential(
            # First conv: pair price/volume
            ConvBlock(ConvBlockConfig(
                in_channels=1,
                out_channels=filters,
                kernel_size=(1, 2),
                stride=(1, 2),
                activation=ActivationType.LEAKY_RELU,
                leaky_relu_slope=leaky_relu_slope,
            )),
            # Temporal convolutions
            ConvBlock(ConvBlockConfig(
                in_channels=filters,
                out_channels=filters,
                kernel_size=(4, 1),
                activation=ActivationType.LEAKY_RELU,
                leaky_relu_slope=leaky_relu_slope,
            )),
            ConvBlock(ConvBlockConfig(
                in_channels=filters,
                out_channels=filters,
                kernel_size=(4, 1),
                activation=ActivationType.LEAKY_RELU,
                leaky_relu_slope=leaky_relu_slope,
            )),
        )
        
        # =====================================================================
        # Block 2: Spatial Reduction
        # Further combines features across LOB levels
        # =====================================================================
        # Conv(1×2, s=1×2): 20 → 10 features
        # Conv(4×1): Temporal convolution (with Tanh per paper)
        # Conv(4×1): Temporal convolution (with Tanh per paper)
        self.block2 = nn.Sequential(
            ConvBlock(ConvBlockConfig(
                in_channels=filters,
                out_channels=filters,
                kernel_size=(1, 2),
                stride=(1, 2),
                activation=ActivationType.TANH,  # Paper uses Tanh here
            )),
            ConvBlock(ConvBlockConfig(
                in_channels=filters,
                out_channels=filters,
                kernel_size=(4, 1),
                activation=ActivationType.TANH,
            )),
            ConvBlock(ConvBlockConfig(
                in_channels=filters,
                out_channels=filters,
                kernel_size=(4, 1),
                activation=ActivationType.TANH,
            )),
        )
        
        # =====================================================================
        # Block 3: Level Consolidation
        # Merges all remaining LOB levels into single feature
        # =====================================================================
        # Conv(1×10): 10 → 1 feature (consolidates all levels)
        # Conv(4×1): Temporal convolution
        # Conv(4×1): Temporal convolution
        self.block3 = nn.Sequential(
            ConvBlock(ConvBlockConfig(
                in_channels=filters,
                out_channels=filters,
                kernel_size=(1, num_levels),  # Consolidate all levels
                activation=ActivationType.LEAKY_RELU,
                leaky_relu_slope=leaky_relu_slope,
            )),
            ConvBlock(ConvBlockConfig(
                in_channels=filters,
                out_channels=filters,
                kernel_size=(4, 1),
                activation=ActivationType.LEAKY_RELU,
                leaky_relu_slope=leaky_relu_slope,
            )),
            ConvBlock(ConvBlockConfig(
                in_channels=filters,
                out_channels=filters,
                kernel_size=(4, 1),
                activation=ActivationType.LEAKY_RELU,
                leaky_relu_slope=leaky_relu_slope,
            )),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all convolutional blocks.
        
        Args:
            x: Input [batch, 1, seq_len, 4*num_levels]
        
        Returns:
            Output [batch, filters, seq_len', 1]
        
        Dimension trace (for seq_len=100, num_levels=10):
            Input:  [B, 1, 100, 40]
            Block1: [B, 32, 94, 20]  (100-6=94 from 4×1 convs)
            Block2: [B, 32, 88, 10]  (94-6=88)
            Block3: [B, 32, 82, 1]   (88-6=82)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class DeepLOB(BaseModel):
    """
    DeepLOB model for LOB price movement prediction.
    
    Implements the full architecture from Zhang et al. (2019):
    - Convolutional encoder for spatial feature extraction
    - Inception module for multi-scale temporal patterns
    - LSTM for temporal dynamics
    - Linear classifier
    
    Args:
        config: DeepLOBConfig with model hyperparameters
    
    Input:
        x: Tensor of shape [batch, seq_len, num_features]
           - benchmark mode: [batch, 100, 40] (10 levels × 4 features)
           - extended mode: [batch, 100, 98] (full feature set)
    
    Output:
        logits: Tensor of shape [batch, num_classes]
        
        Apply softmax for probabilities:
            probs = F.softmax(logits, dim=-1)
        
        Or use with CrossEntropyLoss directly (applies log_softmax internally)
    
    Example:
        >>> config = DeepLOBConfig(mode="benchmark")
        >>> model = DeepLOB(config)
        >>> x = torch.randn(32, 100, 40)  # [batch, seq, features]
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([32, 3])
    
    Note:
        The original paper returns softmax probabilities. We return logits
        for compatibility with PyTorch's CrossEntropyLoss. The paper's
        softmax output can be obtained with F.softmax(logits, dim=-1).
    """
    
    def __init__(self, config: Optional[DeepLOBConfig] = None):
        config = config or DeepLOBConfig()
        super().__init__(config)
        
        self._config: DeepLOBConfig = config
        
        # Feature rearrangement (GROUPED → FI2010 if needed)
        if config.feature_layout == FeatureLayout.GROUPED:
            self.feature_rearrange = FeatureRearrangement(
                source=FeatureLayout.GROUPED,
                target=FeatureLayout.FI2010,
                num_levels=config.num_levels,
            )
        else:
            self.feature_rearrange = nn.Identity()
        
        # Convolutional encoder
        self.conv_encoder = ConvolutionalEncoder(
            num_levels=config.num_levels,
            filters=config.conv_filters,
            leaky_relu_slope=config.leaky_relu_slope,
        )
        
        # Inception module
        self.inception = InceptionModule(InceptionConfig(
            in_channels=config.conv_filters,
            branch_filters=config.inception_filters,
            short_kernel=3,
            medium_kernel=5,
            pool_kernel=3,
            activation=ActivationType.LEAKY_RELU,
            leaky_relu_slope=config.leaky_relu_slope,
        ))
        
        # LSTM encoder
        # Input: inception output = 3 × inception_filters = 192
        lstm_input_size = config.inception_filters * 3
        self.lstm = TemporalEncoder(TemporalEncoderConfig(
            input_size=lstm_input_size,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            dropout=config.dropout,
            bidirectional=False,  # Paper uses unidirectional
            cell_type="lstm",
        ))
        
        # Classification head
        self.classifier = nn.Linear(config.lstm_hidden, config.num_classes)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, num_features]
               For benchmark mode: [batch, 100, 40]
        
        Returns:
            Logits [batch, num_classes]
        
        Shape trace (benchmark mode, batch=32):
            Input:          [32, 100, 40]
            Rearranged:     [32, 100, 40] (FI2010 layout)
            + Channel:      [32, 1, 100, 40]
            Conv encoder:   [32, 32, 82, 1]
            Inception:      [32, 192, 82, 1]
            Reshape:        [32, 82, 192]
            LSTM:           [32, 64]
            Classifier:     [32, 3]
        """
        batch_size = x.size(0)
        
        # Step 1: Feature rearrangement (if needed)
        x = self.feature_rearrange(x)  # [B, T, F]
        
        # Step 2: Add channel dimension for 2D convolution
        x = add_channel_dim(x)  # [B, 1, T, F]
        
        # Step 3: Convolutional encoder
        x = self.conv_encoder(x)  # [B, C, T', 1]
        
        # Step 4: Inception module
        x = self.inception(x)  # [B, 192, T', 1]
        
        # Step 5: Reshape for LSTM
        # [B, C, T', 1] -> [B, T', C]
        x = x.squeeze(-1).permute(0, 2, 1)  # [B, T', 192]
        
        # Step 6: LSTM encoder (returns last hidden state)
        x = self.lstm(x)  # [B, 64]
        
        # Step 7: Classification
        logits = self.classifier(x)  # [B, num_classes]
        
        return logits
    
    @property
    def name(self) -> str:
        """Model name for logging."""
        cfg = self._config
        return f"DeepLOB-{cfg.mode}-{cfg.conv_filters}f-{cfg.lstm_hidden}h"
    
    @property
    def config(self) -> DeepLOBConfig:
        """Model configuration."""
        return self._config


def create_deeplob(
    mode: str = "benchmark",
    num_levels: int = 10,
    sequence_length: int = 100,
    num_classes: int = 3,
    conv_filters: int = 32,
    inception_filters: int = 64,
    lstm_hidden: int = 64,
    dropout: float = 0.0,
) -> DeepLOB:
    """
    Factory function to create DeepLOB model.
    
    Args:
        mode: "benchmark" for exact paper architecture, "extended" for full features
        num_levels: Number of LOB levels
        sequence_length: Input sequence length
        num_classes: Number of output classes
        conv_filters: Filters in conv blocks
        inception_filters: Filters per inception branch
        lstm_hidden: LSTM hidden dimension
        dropout: Dropout probability
    
    Returns:
        DeepLOB model instance
    
    Example:
        >>> model = create_deeplob(mode="benchmark")
        >>> model = create_deeplob(lstm_hidden=128, dropout=0.1)
    """
    config = DeepLOBConfig(
        mode=mode,
        feature_layout=FeatureLayout.GROUPED,  # Default to our format
        num_levels=num_levels,
        sequence_length=sequence_length,
        num_classes=num_classes,
        conv_filters=conv_filters,
        inception_filters=inception_filters,
        lstm_hidden=lstm_hidden,
        lstm_layers=1,
        dropout=dropout,
    )
    return DeepLOB(config)

