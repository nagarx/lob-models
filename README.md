# LOB Models

Deep learning model architectures for Limit Order Book (LOB) price prediction.

## Overview

This library provides modular, well-tested neural network architectures for predicting short-term price movements from limit order book data. The primary model is **DeepLOB**, a CNN-LSTM hybrid architecture from Zhang et al. (2019).

## Installation

```bash
# From source
cd lob-models
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import torch
from lobmodels import DeepLOB, DeepLOBConfig

# Create model with default (benchmark) configuration
config = DeepLOBConfig(mode="benchmark")
model = DeepLOB(config)

# Forward pass
x = torch.randn(32, 100, 40)  # [batch, seq_len, features]
logits = model(x)
print(logits.shape)  # torch.Size([32, 3])

# Get probabilities
probs = torch.softmax(logits, dim=-1)
```

## Model Architecture

### DeepLOB (Zhang et al. 2019)

```
Input [B, T=100, F=40]
    │
    ▼
Feature Rearrangement (GROUPED → FI2010 layout)
    │
    ▼
Convolutional Encoder
├── Block 1: Conv(1×2) → Conv(4×1) × 2  [spatial + temporal]
├── Block 2: Conv(1×2) → Conv(4×1) × 2  [reduction + temporal]
└── Block 3: Conv(1×10) → Conv(4×1) × 2 [consolidate + temporal]
    │
    ▼
Inception Module
├── Branch 1: 1×1 → 3×1 (short-term patterns)
├── Branch 2: 1×1 → 5×1 (medium-term patterns)
└── Branch 3: MaxPool → 1×1 (local max features)
    │
    ▼ [B, 192, T'=82, 1]
LSTM Encoder (64 hidden)
    │
    ▼ [B, 64]
Linear Classifier → [B, 3]
```

## Feature Layouts

The library supports different LOB feature layouts:

| Layout | Description | Features |
|--------|-------------|----------|
| `GROUPED` | Our Rust pipeline output | `[bid_p(L), ask_p(L), bid_s(L), ask_s(L)]` |
| `FI2010` | Original DeepLOB paper | `[p_a, v_a, p_b, v_b] × L levels` |
| `EXTENDED` | Full 98-feature set | LOB + MBO + Signals |

Automatic conversion between layouts is handled by `FeatureRearrangement`.

## Configuration

All model hyperparameters are configurable via dataclasses:

```python
from lobmodels import DeepLOBConfig, FeatureLayout

config = DeepLOBConfig(
    mode="benchmark",           # "benchmark" or "extended"
    feature_layout=FeatureLayout.GROUPED,
    num_levels=10,
    sequence_length=100,
    num_classes=3,
    conv_filters=32,
    inception_filters=64,
    lstm_hidden=64,
    dropout=0.0,
)
```

## Building Blocks

Individual layers are available for custom architectures:

```python
from lobmodels import ConvBlock, InceptionModule, TemporalEncoder
from lobmodels.config import ConvBlockConfig, InceptionConfig, TemporalEncoderConfig

# Convolutional block
conv = ConvBlock(ConvBlockConfig(
    in_channels=1, out_channels=32,
    kernel_size=(1, 2), stride=(1, 2)
))

# Inception module
inception = InceptionModule(InceptionConfig(
    in_channels=32, branch_filters=64
))

# LSTM encoder
lstm = TemporalEncoder(TemporalEncoderConfig(
    input_size=192, hidden_size=64
))
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=lobmodels --cov-report=html

# Run specific test file
pytest tests/test_models/test_deeplob.py -v
```

## Project Structure

```
lob-models/
├── src/lobmodels/
│   ├── config/         # Configuration dataclasses
│   │   └── base.py     # DeepLOBConfig, ConvBlockConfig, etc.
│   ├── layers/         # Building blocks
│   │   ├── activations.py
│   │   ├── conv.py     # ConvBlock, ConvStack
│   │   ├── inception.py # InceptionModule
│   │   └── temporal.py # TemporalEncoder (LSTM/GRU)
│   ├── models/         # Full architectures
│   │   ├── base.py     # BaseModel
│   │   └── deeplob.py  # DeepLOB
│   └── utils/          # Utilities
│       └── feature_layout.py  # Feature rearrangement
└── tests/              # Comprehensive test suite
```

## References

- Zhang, Zohren & Roberts (2019). "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." IEEE Transactions on Signal Processing.
- [Paper](https://arxiv.org/abs/1808.03668) | [Original Code](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)

## License

MIT

