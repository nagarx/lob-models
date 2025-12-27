"""Tests for configuration classes."""

import pytest
import json
from lobmodels.config import (
    BaseConfig,
    ConvBlockConfig,
    InceptionConfig,
    TemporalEncoderConfig,
    DeepLOBConfig,
    ActivationType,
    FeatureLayout,
)


class TestConvBlockConfig:
    """Tests for ConvBlockConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = ConvBlockConfig(in_channels=1, out_channels=32, kernel_size=(1, 2))
        assert config.stride == (1, 1)
        assert config.activation == ActivationType.LEAKY_RELU
        assert config.use_batchnorm is True
        assert config.leaky_relu_slope == 0.01
    
    def test_validation_in_channels(self):
        """Test validation rejects invalid in_channels."""
        with pytest.raises(ValueError, match="in_channels must be >= 1"):
            ConvBlockConfig(in_channels=0, out_channels=32, kernel_size=(1, 2))
    
    def test_validation_out_channels(self):
        """Test validation rejects invalid out_channels."""
        with pytest.raises(ValueError, match="out_channels must be >= 1"):
            ConvBlockConfig(in_channels=1, out_channels=0, kernel_size=(1, 2))
    
    def test_validation_kernel_size(self):
        """Test validation rejects invalid kernel_size."""
        with pytest.raises(ValueError, match="kernel_size must be >= 1"):
            ConvBlockConfig(in_channels=1, out_channels=32, kernel_size=(0, 2))
    
    def test_validation_leaky_relu_slope(self):
        """Test validation rejects invalid leaky_relu_slope."""
        with pytest.raises(ValueError, match="leaky_relu_slope must be in"):
            ConvBlockConfig(in_channels=1, out_channels=32, kernel_size=(1, 2), leaky_relu_slope=1.5)
    
    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = ConvBlockConfig(in_channels=1, out_channels=32, kernel_size=(1, 2))
        d = config.to_dict()
        assert d['in_channels'] == 1
        assert d['out_channels'] == 32
        assert d['kernel_size'] == (1, 2)


class TestInceptionConfig:
    """Tests for InceptionConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = InceptionConfig(in_channels=32)
        assert config.branch_filters == 64
        assert config.short_kernel == 3
        assert config.medium_kernel == 5
        assert config.pool_kernel == 3
    
    def test_out_channels_property(self):
        """Test out_channels = 3 × branch_filters."""
        config = InceptionConfig(in_channels=32, branch_filters=64)
        assert config.out_channels == 192  # 64 × 3
        
        config = InceptionConfig(in_channels=32, branch_filters=32)
        assert config.out_channels == 96  # 32 × 3
    
    def test_validation_in_channels(self):
        """Test validation rejects invalid in_channels."""
        with pytest.raises(ValueError, match="in_channels must be >= 1"):
            InceptionConfig(in_channels=0)


class TestTemporalEncoderConfig:
    """Tests for TemporalEncoderConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TemporalEncoderConfig(input_size=192)
        assert config.hidden_size == 64
        assert config.num_layers == 1
        assert config.dropout == 0.0
        assert config.bidirectional is False
        assert config.cell_type == "lstm"
    
    def test_output_size_unidirectional(self):
        """Test output_size for unidirectional encoder."""
        config = TemporalEncoderConfig(input_size=192, hidden_size=64, bidirectional=False)
        assert config.output_size == 64
    
    def test_output_size_bidirectional(self):
        """Test output_size for bidirectional encoder."""
        config = TemporalEncoderConfig(input_size=192, hidden_size=64, bidirectional=True)
        assert config.output_size == 128  # 64 × 2
    
    def test_validation_cell_type(self):
        """Test validation rejects invalid cell_type."""
        with pytest.raises(ValueError, match="cell_type must be"):
            TemporalEncoderConfig(input_size=192, cell_type="rnn")


class TestDeepLOBConfig:
    """Tests for DeepLOBConfig."""
    
    def test_default_benchmark_mode(self):
        """Test default configuration is benchmark mode."""
        config = DeepLOBConfig()
        assert config.mode == "benchmark"
        assert config.feature_layout == FeatureLayout.GROUPED
        assert config.num_levels == 10
        assert config.sequence_length == 100
        assert config.num_classes == 3
        assert config.conv_filters == 32
        assert config.inception_filters == 64
        assert config.lstm_hidden == 64
        assert config.dropout == 0.0
    
    def test_input_features_benchmark(self):
        """Test input_features for benchmark mode."""
        config = DeepLOBConfig(mode="benchmark", num_levels=10)
        assert config.input_features == 40  # 10 × 4
    
    def test_input_features_extended(self):
        """Test input_features for extended mode."""
        config = DeepLOBConfig(mode="extended", feature_layout=FeatureLayout.EXTENDED)
        assert config.input_features == 98
    
    def test_inception_output_channels(self):
        """Test inception_output_channels property."""
        config = DeepLOBConfig(inception_filters=64)
        assert config.inception_output_channels == 192  # 64 × 3
    
    def test_validation_mode(self):
        """Test validation rejects invalid mode."""
        with pytest.raises(ValueError, match="mode must be"):
            DeepLOBConfig(mode="invalid")
    
    def test_validation_benchmark_extended_conflict(self):
        """Test benchmark mode rejects EXTENDED layout."""
        with pytest.raises(ValueError, match="benchmark mode requires"):
            DeepLOBConfig(mode="benchmark", feature_layout=FeatureLayout.EXTENDED)
    
    def test_to_json(self):
        """Test JSON serialization."""
        config = DeepLOBConfig()
        json_str = config.to_json()
        d = json.loads(json_str)
        assert d['mode'] == 'benchmark'
        assert d['num_levels'] == 10

