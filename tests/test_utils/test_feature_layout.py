"""Tests for feature layout transformation utilities."""

import pytest
import torch
from lobmodels.config import FeatureLayout
from lobmodels.utils.feature_layout import (
    rearrange_grouped_to_fi2010,
    rearrange_fi2010_to_grouped,
    FeatureRearrangement,
    add_channel_dim,
    remove_channel_dim,
)


class TestRearrangeGroupedToFI2010:
    """Tests for GROUPED → FI2010 transformation."""
    
    def test_output_shape(self):
        """Test output shape is preserved."""
        x = torch.randn(32, 100, 40)
        y = rearrange_grouped_to_fi2010(x, num_levels=10)
        assert y.shape == x.shape
    
    def test_correct_mapping(self):
        """Test features are correctly rearranged."""
        # Create input with known values
        # GROUPED: [bid_p(10), ask_p(10), bid_s(10), ask_s(10)]
        num_levels = 10
        x = torch.zeros(1, 1, 40)
        
        # Set known values for level 0
        x[0, 0, 0] = 100.0   # bid_price_L0
        x[0, 0, 10] = 101.0  # ask_price_L0
        x[0, 0, 20] = 50.0   # bid_size_L0
        x[0, 0, 30] = 60.0   # ask_size_L0
        
        y = rearrange_grouped_to_fi2010(x, num_levels)
        
        # FI2010: [p_ask_L0, v_ask_L0, p_bid_L0, v_bid_L0, ...]
        assert y[0, 0, 0].item() == 101.0   # p_ask_L0
        assert y[0, 0, 1].item() == 60.0    # v_ask_L0
        assert y[0, 0, 2].item() == 100.0   # p_bid_L0
        assert y[0, 0, 3].item() == 50.0    # v_bid_L0
    
    def test_all_levels_mapped(self):
        """Test all levels are correctly mapped."""
        num_levels = 10
        x = torch.zeros(1, 1, 40)
        
        # Set unique values for each feature
        for i in range(40):
            x[0, 0, i] = float(i)
        
        y = rearrange_grouped_to_fi2010(x, num_levels)
        
        # Check level 5 (index 20-23 in FI2010 format)
        level = 5
        fi2010_base = level * 4
        
        # Expected FI2010 indices for level 5:
        # p_ask_L5 = ask_prices[5] = index 15 in GROUPED
        # v_ask_L5 = ask_sizes[5] = index 35 in GROUPED
        # p_bid_L5 = bid_prices[5] = index 5 in GROUPED
        # v_bid_L5 = bid_sizes[5] = index 25 in GROUPED
        assert y[0, 0, fi2010_base + 0].item() == 15.0   # p_ask_L5
        assert y[0, 0, fi2010_base + 1].item() == 35.0   # v_ask_L5
        assert y[0, 0, fi2010_base + 2].item() == 5.0    # p_bid_L5
        assert y[0, 0, fi2010_base + 3].item() == 25.0   # v_bid_L5
    
    def test_invalid_feature_count(self):
        """Test error for wrong number of features."""
        x = torch.randn(1, 1, 50)  # Wrong: should be 40
        with pytest.raises(ValueError, match="Expected 40 features"):
            rearrange_grouped_to_fi2010(x, num_levels=10)
    
    def test_batch_processing(self):
        """Test batch processing works correctly."""
        x = torch.randn(64, 100, 40)
        y = rearrange_grouped_to_fi2010(x, num_levels=10)
        assert y.shape == (64, 100, 40)


class TestRearrangeFI2010ToGrouped:
    """Tests for FI2010 → GROUPED transformation."""
    
    def test_output_shape(self):
        """Test output shape is preserved."""
        x = torch.randn(32, 100, 40)
        y = rearrange_fi2010_to_grouped(x, num_levels=10)
        assert y.shape == x.shape
    
    def test_inverse_operation(self):
        """Test FI2010→GROUPED is inverse of GROUPED→FI2010."""
        x_original = torch.randn(8, 50, 40)
        
        # GROUPED → FI2010 → GROUPED should be identity
        x_fi2010 = rearrange_grouped_to_fi2010(x_original, num_levels=10)
        x_recovered = rearrange_fi2010_to_grouped(x_fi2010, num_levels=10)
        
        assert torch.allclose(x_original, x_recovered)
    
    def test_correct_mapping(self):
        """Test features are correctly rearranged back."""
        # Create FI2010 input with known values for level 0
        num_levels = 10
        x = torch.zeros(1, 1, 40)
        
        # FI2010: [p_ask_L0, v_ask_L0, p_bid_L0, v_bid_L0, ...]
        x[0, 0, 0] = 101.0  # p_ask_L0
        x[0, 0, 1] = 60.0   # v_ask_L0
        x[0, 0, 2] = 100.0  # p_bid_L0
        x[0, 0, 3] = 50.0   # v_bid_L0
        
        y = rearrange_fi2010_to_grouped(x, num_levels)
        
        # GROUPED: [bid_p(10), ask_p(10), bid_s(10), ask_s(10)]
        assert y[0, 0, 0].item() == 100.0   # bid_price_L0
        assert y[0, 0, 10].item() == 101.0  # ask_price_L0
        assert y[0, 0, 20].item() == 50.0   # bid_size_L0
        assert y[0, 0, 30].item() == 60.0   # ask_size_L0


class TestFeatureRearrangement:
    """Tests for FeatureRearrangement module."""
    
    def test_grouped_to_fi2010(self):
        """Test module converts GROUPED to FI2010."""
        module = FeatureRearrangement(
            source=FeatureLayout.GROUPED,
            target=FeatureLayout.FI2010,
            num_levels=10,
        )
        
        x = torch.randn(8, 100, 40)
        y = module(x)
        
        # Verify it's equivalent to function call
        expected = rearrange_grouped_to_fi2010(x, num_levels=10)
        assert torch.allclose(y, expected)
    
    def test_identity_same_layout(self):
        """Test identity when source == target."""
        module = FeatureRearrangement(
            source=FeatureLayout.GROUPED,
            target=FeatureLayout.GROUPED,
            num_levels=10,
        )
        
        x = torch.randn(8, 100, 40)
        y = module(x)
        
        assert torch.allclose(x, y)
    
    def test_extended_to_other_raises(self):
        """Test EXTENDED cannot be converted to other layouts."""
        with pytest.raises(ValueError, match="Cannot convert from EXTENDED"):
            FeatureRearrangement(
                source=FeatureLayout.EXTENDED,
                target=FeatureLayout.FI2010,
                num_levels=10,
            )
    
    def test_extra_repr(self):
        """Test string representation."""
        module = FeatureRearrangement(
            source=FeatureLayout.GROUPED,
            target=FeatureLayout.FI2010,
            num_levels=10,
        )
        repr_str = module.extra_repr()
        assert "grouped" in repr_str
        assert "fi2010" in repr_str


class TestAddRemoveChannelDim:
    """Tests for channel dimension helpers."""
    
    def test_add_channel_dim(self):
        """Test adding channel dimension."""
        x = torch.randn(32, 100, 40)  # [batch, seq, feat]
        y = add_channel_dim(x)
        assert y.shape == (32, 1, 100, 40)  # [batch, 1, seq, feat]
    
    def test_add_channel_dim_wrong_dims(self):
        """Test error for wrong number of dimensions."""
        x = torch.randn(32, 100)  # 2D, should be 3D
        with pytest.raises(ValueError, match="Expected 3D tensor"):
            add_channel_dim(x)
    
    def test_remove_channel_dim(self):
        """Test removing channel dimension."""
        x = torch.randn(32, 64, 82, 1)  # [batch, channels, seq, 1]
        y = remove_channel_dim(x)
        assert y.shape == (32, 82, 64)  # [batch, seq, channels]
    
    def test_remove_channel_dim_wrong_last(self):
        """Test error when last dim is not 1."""
        x = torch.randn(32, 64, 82, 5)  # Last dim is 5, not 1
        with pytest.raises(ValueError, match="Expected last dim to be 1"):
            remove_channel_dim(x)

