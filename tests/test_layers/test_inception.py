"""Tests for Inception module."""

import pytest
import torch
from lobmodels.config import InceptionConfig, ActivationType
from lobmodels.layers.inception import InceptionModule, InceptionBranch, InceptionPoolBranch


class TestInceptionBranch:
    """Tests for individual Inception branches."""
    
    def test_forward_shape_with_reduce(self):
        """Test forward pass with 1×1 reduction."""
        branch = InceptionBranch(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 1),
            use_1x1_reduce=True,
        )
        
        x = torch.randn(8, 32, 82, 1)
        y = branch(x)
        
        # Same padding preserves spatial dims
        assert y.shape == (8, 64, 82, 1)
    
    def test_forward_shape_without_reduce(self):
        """Test forward pass without 1×1 reduction."""
        branch = InceptionBranch(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 1),
            use_1x1_reduce=False,
        )
        
        x = torch.randn(8, 32, 82, 1)
        y = branch(x)
        
        assert y.shape == (8, 64, 82, 1)


class TestInceptionPoolBranch:
    """Tests for Inception pooling branch."""
    
    def test_forward_shape(self):
        """Test forward pass preserves spatial dimensions."""
        branch = InceptionPoolBranch(
            in_channels=32,
            out_channels=64,
            pool_kernel=3,
        )
        
        x = torch.randn(8, 32, 82, 1)
        y = branch(x)
        
        # MaxPool with stride=1 and same padding preserves dims
        assert y.shape == (8, 64, 82, 1)
    
    def test_max_pooling_behavior(self):
        """Test that max pooling takes maximum value."""
        branch = InceptionPoolBranch(
            in_channels=1,
            out_channels=1,
            pool_kernel=3,
            use_batchnorm=False,
        )
        
        # Set conv weights to identity
        branch.conv[0].weight.data.fill_(1.0)
        branch.conv[0].bias.data.fill_(0.0)
        
        # Create input with known values
        x = torch.zeros(1, 1, 5, 1)
        x[0, 0, 2, 0] = 10.0  # Spike in the middle
        
        y = branch(x)
        
        # MaxPool(3) should spread the max value
        # With stride=1 and padding=1, center should see max
        assert y[0, 0, 2, 0].item() == pytest.approx(10.0, abs=0.1)


class TestInceptionModule:
    """Tests for full Inception module."""
    
    def test_forward_shape(self):
        """Test forward pass produces correct output shape."""
        config = InceptionConfig(
            in_channels=32,
            branch_filters=64,
            short_kernel=3,
            medium_kernel=5,
            pool_kernel=3,
        )
        module = InceptionModule(config)
        
        x = torch.randn(8, 32, 82, 1)
        y = module(x)
        
        # Output: 64 × 3 = 192 channels
        assert y.shape == (8, 192, 82, 1)
    
    def test_output_channels_matches_config(self):
        """Test output channels equals config.out_channels."""
        config = InceptionConfig(in_channels=32, branch_filters=32)
        module = InceptionModule(config)
        
        x = torch.randn(4, 32, 50, 1)
        y = module(x)
        
        assert y.shape[1] == config.out_channels  # 32 × 3 = 96
    
    def test_gradient_flow_all_branches(self):
        """Test gradients flow through all three branches."""
        config = InceptionConfig(in_channels=32, branch_filters=64)
        module = InceptionModule(config)
        
        x = torch.randn(4, 32, 82, 1, requires_grad=True)
        y = module(x)
        loss = y.sum()
        loss.backward()
        
        # Check gradients exist in all branches
        assert module.branch_short.branch[0].weight.grad is not None
        assert module.branch_medium.branch[0].weight.grad is not None
        assert module.branch_pool.conv[0].weight.grad is not None
    
    def test_different_kernel_sizes(self):
        """Test with different kernel configurations."""
        config = InceptionConfig(
            in_channels=32,
            branch_filters=64,
            short_kernel=2,
            medium_kernel=7,
            pool_kernel=5,
        )
        module = InceptionModule(config)
        
        x = torch.randn(4, 32, 100, 1)
        y = module(x)
        
        # Should still work with same padding
        assert y.shape == (4, 192, 100, 1)
    
    def test_determinism(self):
        """Test same input gives same output (determinism)."""
        torch.manual_seed(42)
        config = InceptionConfig(in_channels=32, branch_filters=64)
        module = InceptionModule(config)
        module.eval()
        
        x = torch.randn(2, 32, 50, 1)
        
        y1 = module(x)
        y2 = module(x)
        
        assert torch.allclose(y1, y2)

