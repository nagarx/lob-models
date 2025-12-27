"""
Base configuration classes for lob-models.

Design Principles (RULE.md):
- All hyperparameters are explicit and configurable (§4)
- Validation at construction time (§7)
- Serializable for experiment tracking (§4)

Configuration Hierarchy:
    BaseConfig
    ├── ConvBlockConfig      (single conv block)
    ├── InceptionConfig      (inception module)
    ├── TemporalEncoderConfig (LSTM/GRU)
    └── DeepLOBConfig        (full model)
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Tuple, List, Optional, Literal
import json


# =============================================================================
# Enums
# =============================================================================


class ActivationType(str, Enum):
    """Supported activation functions."""
    
    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    TANH = "tanh"
    GELU = "gelu"
    NONE = "none"


class FeatureLayout(str, Enum):
    """
    Feature layout formats for LOB data.
    
    FI2010: Original DeepLOB paper format
        [p_ask_L0, v_ask_L0, p_bid_L0, v_bid_L0, ..., p_ask_L9, v_ask_L9, p_bid_L9, v_bid_L9]
        Total: 40 features (4 per level × 10 levels)
        Order within level: [price_ask, volume_ask, price_bid, volume_bid]
    
    GROUPED: Our Rust pipeline format
        [bid_prices(10), ask_prices(10), bid_sizes(10), ask_sizes(10)]
        Total: 40 features (grouped by type)
        Order: [bid_p_L0..L9, ask_p_L0..L9, bid_s_L0..L9, ask_s_L0..L9]
    
    EXTENDED: Our full 98-feature format
        [LOB(40), Derived(8), MBO(36), Signals(14)]
        Includes rich MBO and trading signal features
    """
    
    FI2010 = "fi2010"
    GROUPED = "grouped"
    EXTENDED = "extended"


# =============================================================================
# Base Configuration
# =============================================================================


@dataclass
class BaseConfig:
    """
    Base configuration class with common utilities.
    
    All config classes inherit from this and gain:
    - Serialization to dict/JSON
    - Validation hook
    - String representation
    """
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate configuration values.
        
        Override in subclasses to add specific validation.
        Should raise ValueError with descriptive message on failure.
        """
        pass
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, d: dict) -> "BaseConfig":
        """Create configuration from dictionary."""
        return cls(**d)


# =============================================================================
# Layer Configurations
# =============================================================================


@dataclass
class ConvBlockConfig(BaseConfig):
    """
    Configuration for a single convolutional block.
    
    A ConvBlock consists of:
        Conv2d → Activation → BatchNorm (optional)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels (filters)
        kernel_size: Convolution kernel size (height, width)
        stride: Convolution stride (height, width)
        padding: Padding mode or explicit padding
        activation: Activation function type
        use_batchnorm: Whether to apply batch normalization
        leaky_relu_slope: Negative slope for LeakyReLU (if used)
    
    Example:
        >>> config = ConvBlockConfig(
        ...     in_channels=1, out_channels=32,
        ...     kernel_size=(1, 2), stride=(1, 2)
        ... )
    """
    
    in_channels: int
    out_channels: int
    kernel_size: Tuple[int, int]
    stride: Tuple[int, int] = (1, 1)
    padding: str | Tuple[int, int] = (0, 0)
    activation: ActivationType = ActivationType.LEAKY_RELU
    use_batchnorm: bool = True
    leaky_relu_slope: float = 0.01
    
    def validate(self) -> None:
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {self.in_channels}")
        if self.out_channels < 1:
            raise ValueError(f"out_channels must be >= 1, got {self.out_channels}")
        if any(k < 1 for k in self.kernel_size):
            raise ValueError(f"kernel_size must be >= 1, got {self.kernel_size}")
        if any(s < 1 for s in self.stride):
            raise ValueError(f"stride must be >= 1, got {self.stride}")
        if not 0 <= self.leaky_relu_slope < 1:
            raise ValueError(f"leaky_relu_slope must be in [0, 1), got {self.leaky_relu_slope}")


@dataclass
class InceptionConfig(BaseConfig):
    """
    Configuration for Inception Module.
    
    The Inception Module extracts multi-scale temporal patterns via
    parallel branches with different receptive fields.
    
    Architecture (Zhang et al. 2019, Fig. 4):
        Branch 1: 1×1 conv → 3×1 conv (short-term patterns)
        Branch 2: 1×1 conv → 5×1 conv (medium-term patterns)
        Branch 3: MaxPool(3×1) → 1×1 conv (local max features)
        
        All branches are concatenated along channel dimension.
    
    Args:
        in_channels: Number of input channels
        branch_filters: Number of filters per branch (output is 3× this)
        short_kernel: Kernel size for short-term branch (default: 3)
        medium_kernel: Kernel size for medium-term branch (default: 5)
        pool_kernel: Kernel size for max pooling branch (default: 3)
        activation: Activation function
        use_batchnorm: Whether to apply batch normalization
        leaky_relu_slope: Negative slope for LeakyReLU
    
    Output channels: branch_filters × 3
    """
    
    in_channels: int
    branch_filters: int = 64
    short_kernel: int = 3
    medium_kernel: int = 5
    pool_kernel: int = 3
    activation: ActivationType = ActivationType.LEAKY_RELU
    use_batchnorm: bool = True
    leaky_relu_slope: float = 0.01
    
    def validate(self) -> None:
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {self.in_channels}")
        if self.branch_filters < 1:
            raise ValueError(f"branch_filters must be >= 1, got {self.branch_filters}")
        if self.short_kernel < 1:
            raise ValueError(f"short_kernel must be >= 1, got {self.short_kernel}")
        if self.medium_kernel < 1:
            raise ValueError(f"medium_kernel must be >= 1, got {self.medium_kernel}")
    
    @property
    def out_channels(self) -> int:
        """Total output channels (3 branches concatenated)."""
        return self.branch_filters * 3


@dataclass
class TemporalEncoderConfig(BaseConfig):
    """
    Configuration for temporal sequence encoder (LSTM/GRU).
    
    Encodes the temporal dynamics of extracted features.
    
    Args:
        input_size: Size of input features at each timestep
        hidden_size: Hidden dimension of recurrent layers
        num_layers: Number of stacked recurrent layers
        dropout: Dropout probability between layers (0 if num_layers=1)
        bidirectional: Use bidirectional encoding
        cell_type: Type of recurrent cell ("lstm" or "gru")
    
    Output size: hidden_size × 2 if bidirectional else hidden_size
    """
    
    input_size: int
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    bidirectional: bool = False
    cell_type: Literal["lstm", "gru"] = "lstm"
    
    def validate(self) -> None:
        if self.input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {self.input_size}")
        if self.hidden_size < 1:
            raise ValueError(f"hidden_size must be >= 1, got {self.hidden_size}")
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.cell_type not in ("lstm", "gru"):
            raise ValueError(f"cell_type must be 'lstm' or 'gru', got {self.cell_type}")
    
    @property
    def output_size(self) -> int:
        """Output dimension of encoder."""
        return self.hidden_size * (2 if self.bidirectional else 1)


# =============================================================================
# Model Configurations
# =============================================================================


@dataclass
class DeepLOBConfig(BaseConfig):
    """
    Configuration for DeepLOB model.
    
    Reference:
        Zhang, Zohren & Roberts (2019). "DeepLOB: Deep Convolutional Neural 
        Networks for Limit Order Books." IEEE Trans. Signal Processing.
    
    Modes:
        benchmark: Exact paper architecture for 40 LOB features
        extended: Adapted architecture for 98 features (LOB + MBO + Signals)
    
    Architecture Overview:
        Input → [FeatureRearrangement] → ConvEncoder → Inception → LSTM → Linear
    
    Args:
        mode: "benchmark" for exact paper, "extended" for full features
        feature_layout: Input feature layout format
        num_levels: Number of LOB levels (default: 10)
        sequence_length: Input sequence length (default: 100)
        num_classes: Number of output classes (default: 3)
        
        conv_filters: Filters in convolutional blocks (default: 32)
        inception_filters: Filters per inception branch (default: 64)
        lstm_hidden: LSTM hidden dimension (default: 64)
        lstm_layers: Number of LSTM layers (default: 1)
        
        dropout: Dropout probability (default: 0.0, paper uses none)
        leaky_relu_slope: LeakyReLU negative slope (default: 0.01)
    
    Example:
        >>> # Benchmark mode (exact paper architecture)
        >>> config = DeepLOBConfig(mode="benchmark")
        >>> model = DeepLOB(config)
        >>> 
        >>> # Extended mode (all 98 features)
        >>> config = DeepLOBConfig(mode="extended", feature_layout=FeatureLayout.EXTENDED)
        >>> model = DeepLOB(config)
    """
    
    # Mode
    mode: Literal["benchmark", "extended"] = "benchmark"
    feature_layout: FeatureLayout = FeatureLayout.GROUPED
    
    # Input dimensions
    num_levels: int = 10
    sequence_length: int = 100
    num_classes: int = 3
    
    # Convolutional encoder
    conv_filters: int = 32
    
    # Inception module
    inception_filters: int = 64
    
    # LSTM
    lstm_hidden: int = 64
    lstm_layers: int = 1
    
    # Regularization
    dropout: float = 0.0
    
    # Activation parameters
    leaky_relu_slope: float = 0.01
    
    def validate(self) -> None:
        if self.mode not in ("benchmark", "extended"):
            raise ValueError(f"mode must be 'benchmark' or 'extended', got {self.mode}")
        if self.num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {self.num_levels}")
        if self.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {self.sequence_length}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if self.conv_filters < 1:
            raise ValueError(f"conv_filters must be >= 1, got {self.conv_filters}")
        if self.inception_filters < 1:
            raise ValueError(f"inception_filters must be >= 1, got {self.inception_filters}")
        if self.lstm_hidden < 1:
            raise ValueError(f"lstm_hidden must be >= 1, got {self.lstm_hidden}")
        if not 0 <= self.dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        
        # Mode-specific validation
        if self.mode == "benchmark":
            if self.feature_layout == FeatureLayout.EXTENDED:
                raise ValueError(
                    "benchmark mode requires FI2010 or GROUPED layout, not EXTENDED. "
                    "Use mode='extended' for 98-feature input."
                )
    
    @property
    def input_features(self) -> int:
        """Number of input features based on mode."""
        if self.mode == "benchmark":
            return self.num_levels * 4  # 40 for 10 levels
        else:
            return 98  # Full feature set
    
    @property
    def inception_output_channels(self) -> int:
        """Output channels from inception module (3 branches)."""
        return self.inception_filters * 3

