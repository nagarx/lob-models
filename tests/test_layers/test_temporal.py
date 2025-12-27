"""Tests for temporal encoders (LSTM/GRU)."""

import pytest
import torch
from lobmodels.config import TemporalEncoderConfig
from lobmodels.layers.temporal import TemporalEncoder


class TestTemporalEncoder:
    """Tests for TemporalEncoder (LSTM/GRU)."""
    
    def test_lstm_forward_shape(self):
        """Test LSTM forward pass produces correct output shape."""
        config = TemporalEncoderConfig(
            input_size=192,
            hidden_size=64,
            num_layers=1,
            bidirectional=False,
            cell_type="lstm",
        )
        encoder = TemporalEncoder(config)
        
        x = torch.randn(32, 82, 192)  # [batch, seq, features]
        y = encoder(x)
        
        assert y.shape == (32, 64)  # [batch, hidden]
    
    def test_gru_forward_shape(self):
        """Test GRU forward pass produces correct output shape."""
        config = TemporalEncoderConfig(
            input_size=192,
            hidden_size=64,
            cell_type="gru",
        )
        encoder = TemporalEncoder(config)
        
        x = torch.randn(32, 82, 192)
        y = encoder(x)
        
        assert y.shape == (32, 64)
    
    def test_bidirectional_output_size(self):
        """Test bidirectional encoder doubles output size."""
        config = TemporalEncoderConfig(
            input_size=192,
            hidden_size=64,
            bidirectional=True,
        )
        encoder = TemporalEncoder(config)
        
        x = torch.randn(16, 50, 192)
        y = encoder(x)
        
        assert y.shape == (16, 128)  # 64 × 2
        assert encoder.output_size == 128
    
    def test_multi_layer(self):
        """Test multi-layer encoder."""
        config = TemporalEncoderConfig(
            input_size=192,
            hidden_size=64,
            num_layers=3,
            dropout=0.1,
        )
        encoder = TemporalEncoder(config)
        
        x = torch.randn(8, 100, 192)
        y = encoder(x)
        
        # Output shape same regardless of num_layers
        assert y.shape == (8, 64)
    
    def test_forward_with_sequence(self):
        """Test forward_with_sequence returns both outputs."""
        config = TemporalEncoderConfig(input_size=192, hidden_size=64)
        encoder = TemporalEncoder(config)
        
        x = torch.randn(8, 50, 192)
        output, hidden = encoder.forward_with_sequence(x)
        
        # output: [batch, seq, hidden × directions]
        assert output.shape == (8, 50, 64)
        # hidden: [batch, hidden × directions]
        assert hidden.shape == (8, 64)
    
    def test_forward_with_sequence_bidirectional(self):
        """Test forward_with_sequence with bidirectional."""
        config = TemporalEncoderConfig(
            input_size=192,
            hidden_size=64,
            bidirectional=True,
        )
        encoder = TemporalEncoder(config)
        
        x = torch.randn(8, 50, 192)
        output, hidden = encoder.forward_with_sequence(x)
        
        # Bidirectional doubles channel dimension
        assert output.shape == (8, 50, 128)
        assert hidden.shape == (8, 128)
    
    def test_custom_initial_states(self):
        """Test with custom initial hidden states."""
        config = TemporalEncoderConfig(input_size=192, hidden_size=64)
        encoder = TemporalEncoder(config)
        
        batch_size = 8
        x = torch.randn(batch_size, 50, 192)
        h0 = torch.zeros(1, batch_size, 64)
        c0 = torch.zeros(1, batch_size, 64)
        
        y = encoder(x, h0=h0, c0=c0)
        
        assert y.shape == (batch_size, 64)
    
    def test_gradient_flow(self):
        """Test gradients flow through encoder."""
        config = TemporalEncoderConfig(input_size=192, hidden_size=64)
        encoder = TemporalEncoder(config)
        
        x = torch.randn(4, 50, 192, requires_grad=True)
        y = encoder(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_determinism_eval_mode(self):
        """Test determinism in eval mode."""
        torch.manual_seed(42)
        config = TemporalEncoderConfig(
            input_size=192,
            hidden_size=64,
            dropout=0.5,  # Dropout should be disabled in eval
        )
        encoder = TemporalEncoder(config)
        encoder.eval()
        
        x = torch.randn(4, 50, 192)
        
        y1 = encoder(x)
        y2 = encoder(x)
        
        assert torch.allclose(y1, y2)
    
    def test_weight_initialization(self):
        """Test weights are properly initialized."""
        config = TemporalEncoderConfig(input_size=192, hidden_size=64)
        encoder = TemporalEncoder(config)
        
        # Check forget gate bias is initialized to 1
        for name, param in encoder.rnn.named_parameters():
            if 'bias_ih' in name:
                n = param.size(0)
                forget_bias = param[n // 4:n // 2]
                # Should be close to 1 (forget gate initialization)
                assert torch.allclose(forget_bias, torch.ones_like(forget_bias))

