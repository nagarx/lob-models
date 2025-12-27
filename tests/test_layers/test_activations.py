"""Tests for activation functions."""

import pytest
import torch
from lobmodels.config import ActivationType
from lobmodels.layers.activations import get_activation


class TestGetActivation:
    """Tests for get_activation utility."""
    
    def test_relu(self):
        """Test ReLU activation."""
        act = get_activation(ActivationType.RELU)
        x = torch.tensor([-1.0, 0.0, 1.0])
        y = act(x)
        
        assert y[0].item() == 0.0
        assert y[1].item() == 0.0
        assert y[2].item() == 1.0
    
    def test_leaky_relu(self):
        """Test LeakyReLU activation."""
        slope = 0.01
        act = get_activation(ActivationType.LEAKY_RELU, leaky_relu_slope=slope)
        x = torch.tensor([-1.0, 0.0, 1.0])
        y = act(x)
        
        assert y[0].item() == pytest.approx(-0.01, abs=1e-5)
        assert y[1].item() == 0.0
        assert y[2].item() == 1.0
    
    def test_leaky_relu_custom_slope(self):
        """Test LeakyReLU with custom slope."""
        slope = 0.2
        act = get_activation(ActivationType.LEAKY_RELU, leaky_relu_slope=slope)
        x = torch.tensor([-1.0])
        y = act(x)
        
        assert y[0].item() == pytest.approx(-0.2, abs=1e-5)
    
    def test_tanh(self):
        """Test Tanh activation."""
        act = get_activation(ActivationType.TANH)
        x = torch.tensor([0.0, 100.0, -100.0])
        y = act(x)
        
        assert y[0].item() == pytest.approx(0.0, abs=1e-5)
        assert y[1].item() == pytest.approx(1.0, abs=1e-3)
        assert y[2].item() == pytest.approx(-1.0, abs=1e-3)
    
    def test_gelu(self):
        """Test GELU activation."""
        act = get_activation(ActivationType.GELU)
        x = torch.tensor([0.0, 1.0, -1.0])
        y = act(x)
        
        # GELU(0) = 0
        assert y[0].item() == pytest.approx(0.0, abs=1e-5)
        # GELU(1) ≈ 0.841
        assert 0.8 < y[1].item() < 0.9
    
    def test_none(self):
        """Test None (identity) activation."""
        act = get_activation(ActivationType.NONE)
        x = torch.tensor([-1.0, 0.0, 1.0])
        y = act(x)
        
        assert torch.allclose(x, y)
    
    def test_string_input(self):
        """Test activation from string."""
        act = get_activation("leaky_relu", leaky_relu_slope=0.01)
        x = torch.tensor([-1.0])
        y = act(x)
        
        assert y[0].item() == pytest.approx(-0.01, abs=1e-5)
    
    def test_invalid_string(self):
        """Test error for invalid activation string."""
        with pytest.raises(ValueError, match="Unknown activation type"):
            get_activation("invalid_activation")

