"""Tests for DeepLOB model."""

import pytest
import torch
from lobmodels.config import DeepLOBConfig, FeatureLayout
from lobmodels.models.deeplob import DeepLOB, ConvolutionalEncoder, create_deeplob


class TestConvolutionalEncoder:
    """Tests for ConvolutionalEncoder."""
    
    def test_forward_shape_default(self):
        """Test forward pass produces expected output shape."""
        encoder = ConvolutionalEncoder(num_levels=10, filters=32)
        
        x = torch.randn(8, 1, 100, 40)  # [batch, 1, seq, features]
        y = encoder(x)
        
        # After 3 blocks of temporal convolutions:
        # Each (4,1) kernel reduces height by 3
        # Block1: 100 - 3 - 3 = 94
        # Block2: 94 - 3 - 3 = 88
        # Block3: 88 - 3 - 3 = 82
        # Width: 40 → 20 → 10 → 1
        assert y.shape == (8, 32, 82, 1)
    
    def test_forward_shape_different_seq_length(self):
        """Test with different sequence lengths."""
        encoder = ConvolutionalEncoder(num_levels=10, filters=32)
        
        # Shorter sequence
        x = torch.randn(4, 1, 50, 40)
        y = encoder(x)
        
        # 50 - 18 = 32
        assert y.shape == (4, 32, 32, 1)
    
    def test_gradient_flow(self):
        """Test gradients flow through encoder."""
        encoder = ConvolutionalEncoder(num_levels=10, filters=32)
        
        x = torch.randn(4, 1, 100, 40, requires_grad=True)
        y = encoder(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None


class TestDeepLOB:
    """Tests for DeepLOB model."""
    
    def test_forward_shape_benchmark(self):
        """Test forward pass with benchmark configuration."""
        config = DeepLOBConfig(mode="benchmark")
        model = DeepLOB(config)
        
        x = torch.randn(32, 100, 40)  # [batch, seq, features]
        logits = model(x)
        
        assert logits.shape == (32, 3)  # [batch, num_classes]
    
    def test_forward_shape_custom_classes(self):
        """Test with custom number of classes."""
        config = DeepLOBConfig(num_classes=5)
        model = DeepLOB(config)
        
        x = torch.randn(16, 100, 40)
        logits = model(x)
        
        assert logits.shape == (16, 5)
    
    def test_forward_shape_custom_hidden(self):
        """Test with custom LSTM hidden size."""
        config = DeepLOBConfig(lstm_hidden=128)
        model = DeepLOB(config)
        
        x = torch.randn(8, 100, 40)
        logits = model(x)
        
        assert logits.shape == (8, 3)
    
    def test_feature_rearrangement_grouped(self):
        """Test GROUPED layout is rearranged to FI2010."""
        config = DeepLOBConfig(feature_layout=FeatureLayout.GROUPED)
        model = DeepLOB(config)
        
        # Model should have rearrangement layer
        assert hasattr(model, 'feature_rearrange')
        
        x = torch.randn(4, 100, 40)
        logits = model(x)
        
        assert logits.shape == (4, 3)
    
    def test_gradient_flow(self):
        """Test gradients flow through entire model."""
        config = DeepLOBConfig()
        model = DeepLOB(config)
        
        x = torch.randn(8, 100, 40, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape
    
    def test_model_name(self):
        """Test model name property."""
        config = DeepLOBConfig(mode="benchmark", conv_filters=32, lstm_hidden=64)
        model = DeepLOB(config)
        
        assert "DeepLOB" in model.name
        assert "benchmark" in model.name
    
    def test_num_parameters(self):
        """Test parameter counting."""
        config = DeepLOBConfig()
        model = DeepLOB(config)
        
        # Should have reasonable number of parameters (~60k as per paper)
        num_params = model.num_parameters
        assert num_params > 0
        assert num_params < 1_000_000  # Less than 1M
    
    def test_get_info(self):
        """Test get_info method."""
        config = DeepLOBConfig()
        model = DeepLOB(config)
        
        info = model.get_info()
        
        assert 'name' in info
        assert 'num_params' in info
        assert 'trainable_params' in info
        assert 'config' in info
    
    def test_freeze_unfreeze(self):
        """Test freeze and unfreeze methods."""
        config = DeepLOBConfig()
        model = DeepLOB(config)
        
        # Initially trainable
        assert model.num_trainable_parameters > 0
        
        # Freeze
        model.freeze()
        assert model.num_trainable_parameters == 0
        
        # Unfreeze
        model.unfreeze()
        assert model.num_trainable_parameters > 0
    
    def test_determinism_eval_mode(self):
        """Test determinism in eval mode."""
        torch.manual_seed(42)
        config = DeepLOBConfig()
        model = DeepLOB(config)
        model.eval()
        
        x = torch.randn(4, 100, 40)
        
        logits1 = model(x)
        logits2 = model(x)
        
        assert torch.allclose(logits1, logits2)
    
    def test_training_step(self):
        """Test a complete training step."""
        config = DeepLOBConfig()
        model = DeepLOB(config)
        model.train()
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        x = torch.randn(8, 100, 40)
        y = torch.randint(0, 3, (8,))  # Random labels
        
        # Forward
        logits = model(x)
        loss = criterion(logits, y)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check loss is valid
        assert not torch.isnan(loss)
        assert loss.item() > 0


class TestCreateDeepLOB:
    """Tests for create_deeplob factory function."""
    
    def test_default_creation(self):
        """Test default model creation."""
        model = create_deeplob()
        
        assert isinstance(model, DeepLOB)
        assert model.config.mode == "benchmark"
    
    def test_custom_parameters(self):
        """Test creation with custom parameters."""
        model = create_deeplob(
            mode="benchmark",
            num_levels=10,
            lstm_hidden=128,
            dropout=0.1,
        )
        
        assert model.config.lstm_hidden == 128
        assert model.config.dropout == 0.1
    
    def test_model_is_functional(self):
        """Test created model works."""
        model = create_deeplob()
        
        x = torch.randn(4, 100, 40)
        logits = model(x)
        
        assert logits.shape == (4, 3)


class TestDeepLOBShapeTrace:
    """
    Detailed tests verifying tensor shapes through the model.
    
    These tests document the expected dimensional flow and serve
    as a specification for the architecture.
    """
    
    @pytest.fixture
    def model(self):
        """Create model for shape tests."""
        return DeepLOB(DeepLOBConfig())
    
    def test_shape_after_rearrangement(self, model):
        """Test shape after feature rearrangement."""
        x = torch.randn(4, 100, 40)
        y = model.feature_rearrange(x)
        assert y.shape == (4, 100, 40)
    
    def test_shape_after_conv_encoder(self, model):
        """Test shape after convolutional encoder."""
        x = torch.randn(4, 1, 100, 40)
        y = model.conv_encoder(x)
        
        # 100 - 18 = 82 (from 6 temporal convolutions of kernel size 4)
        # 40 → 20 → 10 → 1 (from 3 spatial reductions)
        assert y.shape == (4, 32, 82, 1)
    
    def test_shape_after_inception(self, model):
        """Test shape after inception module."""
        x = torch.randn(4, 32, 82, 1)
        y = model.inception(x)
        
        # 3 branches × 64 filters = 192 channels
        assert y.shape == (4, 192, 82, 1)
    
    def test_shape_after_lstm(self, model):
        """Test shape after LSTM encoder."""
        x = torch.randn(4, 82, 192)  # [batch, seq, features]
        y = model.lstm(x)
        
        # LSTM hidden size = 64
        assert y.shape == (4, 64)
    
    def test_full_forward_dimensions_documented(self):
        """
        Document and verify full forward pass dimensions.
        
        This serves as executable documentation of the architecture.
        """
        config = DeepLOBConfig()
        model = DeepLOB(config)
        
        batch_size = 4
        seq_len = 100
        num_features = 40
        
        # Input
        x = torch.randn(batch_size, seq_len, num_features)
        assert x.shape == (4, 100, 40), "Input: [batch, seq, features]"
        
        # After rearrangement (GROUPED → FI2010)
        x_rearranged = model.feature_rearrange(x)
        assert x_rearranged.shape == (4, 100, 40), "Rearranged: [batch, seq, features]"
        
        # Add channel dim
        from lobmodels.utils.feature_layout import add_channel_dim
        x_4d = add_channel_dim(x_rearranged)
        assert x_4d.shape == (4, 1, 100, 40), "4D: [batch, 1, seq, features]"
        
        # After conv encoder
        x_conv = model.conv_encoder(x_4d)
        assert x_conv.shape == (4, 32, 82, 1), "Conv: [batch, filters, seq', 1]"
        
        # After inception
        x_inception = model.inception(x_conv)
        assert x_inception.shape == (4, 192, 82, 1), "Inception: [batch, 192, seq', 1]"
        
        # Reshape for LSTM
        x_lstm_input = x_inception.squeeze(-1).permute(0, 2, 1)
        assert x_lstm_input.shape == (4, 82, 192), "LSTM input: [batch, seq', 192]"
        
        # After LSTM
        x_lstm_out = model.lstm(x_lstm_input)
        assert x_lstm_out.shape == (4, 64), "LSTM output: [batch, hidden]"
        
        # Final logits
        logits = model.classifier(x_lstm_out)
        assert logits.shape == (4, 3), "Logits: [batch, num_classes]"

