"""Tests for convolutional layers."""

import pytest
import torch
from lobmodels.config import ConvBlockConfig, ActivationType
from lobmodels.layers.conv import ConvBlock, ConvStack, compute_output_size


class TestConvBlock:
    """Tests for ConvBlock layer."""
    
    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        config = ConvBlockConfig(
            in_channels=1,
            out_channels=32,
            kernel_size=(1, 2),
            stride=(1, 2),
        )
        block = ConvBlock(config)
        
        x = torch.randn(8, 1, 100, 40)  # [batch, channels, seq, features]
        y = block(x)
        
        # Width: 40 / 2 = 20 (due to stride 2)
        # Height: 100 (kernel 1, stride 1)
        assert y.shape == (8, 32, 100, 20)
    
    def test_forward_temporal_conv(self):
        """Test temporal convolution (height reduction)."""
        config = ConvBlockConfig(
            in_channels=32,
            out_channels=32,
            kernel_size=(4, 1),  # Temporal kernel
        )
        block = ConvBlock(config)
        
        x = torch.randn(8, 32, 100, 20)
        y = block(x)
        
        # Height: 100 - 4 + 1 = 97
        assert y.shape == (8, 32, 97, 20)
    
    def test_leaky_relu_activation(self):
        """Test LeakyReLU activation is applied."""
        config = ConvBlockConfig(
            in_channels=1,
            out_channels=1,  # Single output channel for scalar test
            kernel_size=(1, 1),
            activation=ActivationType.LEAKY_RELU,
            leaky_relu_slope=0.01,
            use_batchnorm=False,  # Disable BN for simpler testing
        )
        block = ConvBlock(config)
        
        # Create input with known negative values
        x = torch.full((1, 1, 1, 1), -1.0)
        block.conv.weight.data.fill_(1.0)
        block.conv.bias.data.fill_(0.0)
        
        y = block(x)
        
        # LeakyReLU should keep negative but scale by 0.01
        # Conv output = -1, LeakyReLU(-1, slope=0.01) = -0.01
        assert y.item() == pytest.approx(-0.01, abs=1e-5)
    
    def test_tanh_activation(self):
        """Test Tanh activation is applied."""
        config = ConvBlockConfig(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, 1),
            activation=ActivationType.TANH,
            use_batchnorm=False,
        )
        block = ConvBlock(config)
        
        x = torch.full((1, 1, 1, 1), 100.0)  # Large positive
        block.conv.weight.data.fill_(1.0)
        block.conv.bias.data.fill_(0.0)
        
        y = block(x)
        
        # Tanh should saturate near 1.0
        assert y.item() == pytest.approx(1.0, abs=0.001)
    
    def test_batchnorm_applied(self):
        """Test batch normalization is applied when enabled."""
        config = ConvBlockConfig(
            in_channels=1,
            out_channels=32,
            kernel_size=(1, 1),
            use_batchnorm=True,
        )
        block = ConvBlock(config)
        
        # BN requires >1 sample for running stats
        x = torch.randn(16, 1, 10, 10)
        y = block(x)
        
        # Should not raise and output should be normalized
        assert y.shape == (16, 32, 10, 10)
    
    def test_gradient_flow(self):
        """Test gradients flow through the block."""
        config = ConvBlockConfig(
            in_channels=1,
            out_channels=32,
            kernel_size=(1, 2),
            stride=(1, 2),
        )
        block = ConvBlock(config)
        
        x = torch.randn(4, 1, 100, 40, requires_grad=True)
        y = block(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape


class TestConvStack:
    """Tests for ConvStack (sequential conv blocks)."""
    
    def test_forward_shape(self):
        """Test forward pass through multiple blocks."""
        configs = [
            ConvBlockConfig(1, 32, (1, 2), (1, 2)),
            ConvBlockConfig(32, 32, (4, 1)),
            ConvBlockConfig(32, 32, (4, 1)),
        ]
        stack = ConvStack(configs)
        
        x = torch.randn(8, 1, 100, 40)
        y = stack(x)
        
        # Width: 40 / 2 = 20
        # Height: 100 - 3 - 3 = 94
        assert y.shape == (8, 32, 94, 20)
    
    def test_empty_configs_raises(self):
        """Test empty config list raises error."""
        with pytest.raises(ValueError, match="configs list cannot be empty"):
            ConvStack([])


class TestComputeOutputSize:
    """Tests for compute_output_size helper."""
    
    def test_stride_reduction(self):
        """Test output size with stride > 1."""
        out = compute_output_size((100, 40), (1, 2), (1, 2), (0, 0))
        assert out == (100, 20)
    
    def test_kernel_reduction(self):
        """Test output size with kernel > 1."""
        out = compute_output_size((100, 20), (4, 1), (1, 1), (0, 0))
        assert out == (97, 20)
    
    def test_same_padding(self):
        """Test 'same' padding preserves size."""
        out = compute_output_size((100, 40), (4, 1), (1, 1), 'same')
        assert out == (100, 40)
    
    def test_valid_padding(self):
        """Test 'valid' padding (no padding)."""
        out = compute_output_size((100, 40), (4, 1), (1, 1), 'valid')
        assert out == (97, 40)

