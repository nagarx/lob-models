"""
Temporal sequence encoders (LSTM/GRU).

The temporal encoder captures long-range dependencies in the feature
sequences extracted by the convolutional layers.

Reference:
    Zhang et al. (2019), Section IV.B.c:
    "In order to capture temporal relationship that exist in the extracted 
    features, we replace the fully connected layers with LSTM units."
    
    Original LSTM: Hochreiter & Schmidhuber (1997)

Architecture:
    Input [batch, seq_len, features] → LSTM → Last hidden state [batch, hidden]

Design Principles (RULE.md):
- Configuration-driven (§4)
- Support both LSTM and GRU (§3 modularity)
- Deterministic given same seed (§6)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from lobmodels.config import TemporalEncoderConfig


class TemporalEncoder(nn.Module):
    """
    LSTM/GRU-based temporal sequence encoder.
    
    Encodes temporal dynamics of extracted features. Uses the final
    hidden state as the sequence representation.
    
    Args:
        config: TemporalEncoderConfig with encoder parameters
    
    Input shape:  [batch, seq_len, input_size]
    Output shape: [batch, output_size]
        where output_size = hidden_size × 2 if bidirectional else hidden_size
    
    Example:
        >>> config = TemporalEncoderConfig(input_size=192, hidden_size=64)
        >>> encoder = TemporalEncoder(config)
        >>> x = torch.randn(32, 82, 192)  # [batch, seq, features]
        >>> out = encoder(x)
        >>> out.shape
        torch.Size([32, 64])
    
    Note:
        The paper uses a single-layer unidirectional LSTM with 64 hidden units.
        We provide options for multi-layer and bidirectional for experimentation.
    """
    
    def __init__(self, config: TemporalEncoderConfig):
        super().__init__()
        self.config = config
        
        # Create recurrent layer
        rnn_class = nn.LSTM if config.cell_type == "lstm" else nn.GRU
        
        self.rnn = rnn_class(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self) -> None:
        """
        Initialize RNN weights.
        
        Uses Xavier initialization for input weights and orthogonal
        initialization for recurrent weights (helps with gradient flow).
        """
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: orthogonal
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases: zero, except forget gate (set to 1)
                param.data.fill_(0)
                if self.config.cell_type == "lstm":
                    # LSTM has 4 gates, forget gate is the second
                    n = param.size(0)
                    param.data[n // 4:n // 2].fill_(1.0)
    
    def forward(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode sequence and return final hidden state.
        
        Args:
            x: Input sequence [batch, seq_len, input_size]
            h0: Optional initial hidden state
            c0: Optional initial cell state (LSTM only)
        
        Returns:
            Final hidden representation [batch, output_size]
        """
        batch_size = x.size(0)
        
        # Initialize hidden states if not provided
        num_directions = 2 if self.config.bidirectional else 1
        h_shape = (self.config.num_layers * num_directions, batch_size, self.config.hidden_size)
        
        if h0 is None:
            h0 = torch.zeros(h_shape, device=x.device, dtype=x.dtype)
        
        # Forward through RNN
        if self.config.cell_type == "lstm":
            if c0 is None:
                c0 = torch.zeros(h_shape, device=x.device, dtype=x.dtype)
            output, (h_n, c_n) = self.rnn(x, (h0, c0))
        else:  # GRU
            output, h_n = self.rnn(x, h0)
        
        # Extract final hidden state
        # h_n shape: [num_layers * num_directions, batch, hidden_size]
        if self.config.bidirectional:
            # Concatenate forward and backward final hidden states
            # Take from last layer: indices [-2] (forward) and [-1] (backward)
            h_forward = h_n[-2, :, :]   # [batch, hidden]
            h_backward = h_n[-1, :, :]  # [batch, hidden]
            hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            # Take final layer's hidden state
            hidden = h_n[-1, :, :]  # [batch, hidden]
        
        return hidden
    
    def forward_with_sequence(
        self,
        x: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequence and return both full output and final hidden.
        
        Useful when attention over all timesteps is needed.
        
        Args:
            x: Input sequence [batch, seq_len, input_size]
            h0: Optional initial hidden state
            c0: Optional initial cell state (LSTM only)
        
        Returns:
            Tuple of:
            - output: Full sequence output [batch, seq_len, hidden × directions]
            - hidden: Final hidden state [batch, output_size]
        """
        batch_size = x.size(0)
        num_directions = 2 if self.config.bidirectional else 1
        h_shape = (self.config.num_layers * num_directions, batch_size, self.config.hidden_size)
        
        if h0 is None:
            h0 = torch.zeros(h_shape, device=x.device, dtype=x.dtype)
        
        if self.config.cell_type == "lstm":
            if c0 is None:
                c0 = torch.zeros(h_shape, device=x.device, dtype=x.dtype)
            output, (h_n, c_n) = self.rnn(x, (h0, c0))
        else:
            output, h_n = self.rnn(x, h0)
        
        # Get final hidden
        if self.config.bidirectional:
            h_forward = h_n[-2, :, :]
            h_backward = h_n[-1, :, :]
            hidden = torch.cat([h_forward, h_backward], dim=1)
        else:
            hidden = h_n[-1, :, :]
        
        return output, hidden
    
    @property
    def output_size(self) -> int:
        """Output dimension of encoder."""
        return self.config.output_size
    
    def extra_repr(self) -> str:
        return (
            f"type={self.config.cell_type.upper()}, "
            f"in={self.config.input_size}, "
            f"hidden={self.config.hidden_size}, "
            f"layers={self.config.num_layers}, "
            f"bidir={self.config.bidirectional}"
        )

