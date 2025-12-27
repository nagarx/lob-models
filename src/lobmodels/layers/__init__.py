"""
Reusable neural network building blocks.

All layers are designed to be:
- Configurable via dataclass configs
- Independently testable
- Composable into larger architectures

Available layers:
- ConvBlock: Configurable Conv2d + Activation + BatchNorm
- InceptionModule: Multi-scale temporal pattern extraction
- TemporalEncoder: LSTM/GRU sequence encoder
"""

from lobmodels.layers.activations import get_activation
from lobmodels.layers.conv import ConvBlock
from lobmodels.layers.inception import InceptionModule
from lobmodels.layers.temporal import TemporalEncoder

__all__ = [
    "get_activation",
    "ConvBlock",
    "InceptionModule",
    "TemporalEncoder",
]

