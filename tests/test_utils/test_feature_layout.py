"""
Tests for feature layout transformation utilities.

Layout Reference (matches Rust pipeline output):
    GROUPED: [ask_prices(L), ask_sizes(L), bid_prices(L), bid_sizes(L)]
             Indices: 0:L = ask_p, L:2L = ask_s, 2L:3L = bid_p, 3L:4L = bid_s
    
    FI2010:  [p_a0, v_a0, p_b0, v_b0, ..., p_aL-1, v_aL-1, p_bL-1, v_bL-1]
             (interleaved per level, ask before bid)

These tests validate:
1. Correct feature mapping between layouts
2. Spread invariant preservation (ask_price > bid_price)
3. Bijective transformation (inverse is identity)
4. Shape preservation across batches
"""

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


class TestGroupedLayout:
    """Tests that verify GROUPED layout assumptions."""
    
    def test_grouped_layout_indices(self):
        """
        Verify GROUPED layout index ranges match Rust pipeline.
        
        Source: feature-extractor-MBO-LOB/CODEBASE.md §4
        """
        num_levels = 10
        
        # GROUPED: [ask_p(0:10), ask_s(10:20), bid_p(20:30), bid_s(30:40)]
        ask_price_start = 0
        ask_price_end = num_levels
        ask_size_start = num_levels
        ask_size_end = 2 * num_levels
        bid_price_start = 2 * num_levels
        bid_price_end = 3 * num_levels
        bid_size_start = 3 * num_levels
        bid_size_end = 4 * num_levels
        
        # Verify no overlap and contiguous
        assert ask_price_end == ask_size_start
        assert ask_size_end == bid_price_start
        assert bid_price_end == bid_size_start
        assert bid_size_end == 40
    
    def test_spread_invariant(self):
        """
        Test spread invariant: ask_price_L0 > bid_price_L0.
        
        This matches the Rust pipeline's verify_raw_spreads() check.
        """
        num_levels = 10
        x = torch.zeros(1, 1, 40)
        
        # Set realistic prices (ask > bid)
        best_ask = 100.05
        best_bid = 100.00
        spread = best_ask - best_bid
        
        # GROUPED layout: ask_price_L0 at index 0, bid_price_L0 at index 20
        x[0, 0, 0] = best_ask   # ask_price_L0
        x[0, 0, 20] = best_bid  # bid_price_L0
        
        # Verify spread invariant
        computed_spread = x[0, 0, 0] - x[0, 0, 20]
        assert computed_spread > 0, f"Spread should be positive, got {computed_spread}"
        assert abs(computed_spread.item() - spread) < 1e-4  # Relaxed tolerance for f32 precision


class TestRearrangeGroupedToFI2010:
    """Tests for GROUPED → FI2010 transformation."""
    
    def test_output_shape(self):
        """Test output shape is preserved."""
        x = torch.randn(32, 100, 40)
        y = rearrange_grouped_to_fi2010(x, num_levels=10)
        assert y.shape == x.shape
    
    def test_correct_mapping_level0(self):
        """
        Test features are correctly rearranged for level 0.
        
        GROUPED: [ask_p(10), ask_s(10), bid_p(10), bid_s(10)]
        FI2010:  [p_a0, v_a0, p_b0, v_b0, ...]
        """
        num_levels = 10
        x = torch.zeros(1, 1, 40)
        
        # Set known values for level 0 in GROUPED layout
        # Rust layout: ask_p[0], ask_s[10], bid_p[20], bid_s[30]
        ask_price_L0 = 101.0
        ask_size_L0 = 60.0
        bid_price_L0 = 100.0
        bid_size_L0 = 50.0
        
        x[0, 0, 0] = ask_price_L0   # ask_price at index 0
        x[0, 0, 10] = ask_size_L0   # ask_size at index 10
        x[0, 0, 20] = bid_price_L0  # bid_price at index 20
        x[0, 0, 30] = bid_size_L0   # bid_size at index 30
        
        y = rearrange_grouped_to_fi2010(x, num_levels)
        
        # FI2010 level 0: [p_ask_L0, v_ask_L0, p_bid_L0, v_bid_L0] at indices 0-3
        assert y[0, 0, 0].item() == ask_price_L0, "p_ask_L0"
        assert y[0, 0, 1].item() == ask_size_L0, "v_ask_L0"
        assert y[0, 0, 2].item() == bid_price_L0, "p_bid_L0"
        assert y[0, 0, 3].item() == bid_size_L0, "v_bid_L0"
    
    def test_all_levels_mapped(self):
        """Test all levels are correctly mapped."""
        num_levels = 10
        x = torch.zeros(1, 1, 40)
        
        # Set unique values for each feature
        for i in range(40):
            x[0, 0, i] = float(i)
        
        y = rearrange_grouped_to_fi2010(x, num_levels)
        
        # Check level 5 (FI2010 indices 20-23)
        level = 5
        fi2010_base = level * 4
        
        # GROUPED indices for level 5:
        # ask_price_L5 = index 5 (0 + 5)
        # ask_size_L5 = index 15 (10 + 5)
        # bid_price_L5 = index 25 (20 + 5)
        # bid_size_L5 = index 35 (30 + 5)
        
        # FI2010 expects: [p_ask, v_ask, p_bid, v_bid] per level
        assert y[0, 0, fi2010_base + 0].item() == 5.0, "p_ask_L5 from index 5"
        assert y[0, 0, fi2010_base + 1].item() == 15.0, "v_ask_L5 from index 15"
        assert y[0, 0, fi2010_base + 2].item() == 25.0, "p_bid_L5 from index 25"
        assert y[0, 0, fi2010_base + 3].item() == 35.0, "v_bid_L5 from index 35"
    
    def test_spread_invariant_preserved(self):
        """Test spread invariant is preserved after transformation."""
        num_levels = 10
        x = torch.zeros(1, 1, 40)
        
        # Set prices with positive spread
        best_ask = 100.05
        best_bid = 100.00
        x[0, 0, 0] = best_ask   # ask_price_L0
        x[0, 0, 20] = best_bid  # bid_price_L0
        
        y = rearrange_grouped_to_fi2010(x, num_levels)
        
        # In FI2010: p_ask_L0 at index 0, p_bid_L0 at index 2
        fi2010_spread = y[0, 0, 0] - y[0, 0, 2]
        assert fi2010_spread > 0, f"FI2010 spread should be positive"
    
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
    
    def test_different_num_levels(self):
        """Test transformation works with different level counts."""
        for num_levels in [5, 8, 10, 15]:
            features = 4 * num_levels
            x = torch.randn(2, 10, features)
            y = rearrange_grouped_to_fi2010(x, num_levels=num_levels)
            assert y.shape == x.shape


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
        
        assert torch.allclose(x_original, x_recovered), "Transformation should be bijective"
    
    def test_inverse_other_direction(self):
        """Test GROUPED→FI2010 is inverse of FI2010→GROUPED."""
        x_original = torch.randn(8, 50, 40)
        
        # FI2010 → GROUPED → FI2010 should be identity
        x_grouped = rearrange_fi2010_to_grouped(x_original, num_levels=10)
        x_recovered = rearrange_grouped_to_fi2010(x_grouped, num_levels=10)
        
        assert torch.allclose(x_original, x_recovered), "Inverse transformation should be bijective"
    
    def test_correct_mapping(self):
        """Test features are correctly rearranged back."""
        num_levels = 10
        x = torch.zeros(1, 1, 40)
        
        # Create FI2010 input with known values for level 0
        # FI2010: [p_ask_L0, v_ask_L0, p_bid_L0, v_bid_L0, ...]
        x[0, 0, 0] = 101.0  # p_ask_L0
        x[0, 0, 1] = 60.0   # v_ask_L0
        x[0, 0, 2] = 100.0  # p_bid_L0
        x[0, 0, 3] = 50.0   # v_bid_L0
        
        y = rearrange_fi2010_to_grouped(x, num_levels)
        
        # GROUPED: [ask_p(10), ask_s(10), bid_p(10), bid_s(10)]
        assert y[0, 0, 0].item() == 101.0, "ask_price_L0 at index 0"
        assert y[0, 0, 10].item() == 60.0, "ask_size_L0 at index 10"
        assert y[0, 0, 20].item() == 100.0, "bid_price_L0 at index 20"
        assert y[0, 0, 30].item() == 50.0, "bid_size_L0 at index 30"


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
    
    def test_fi2010_to_grouped(self):
        """Test module converts FI2010 to GROUPED."""
        module = FeatureRearrangement(
            source=FeatureLayout.FI2010,
            target=FeatureLayout.GROUPED,
            num_levels=10,
        )
        
        x = torch.randn(8, 100, 40)
        y = module(x)
        
        # Verify it's equivalent to function call
        expected = rearrange_fi2010_to_grouped(x, num_levels=10)
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


class TestLayoutConsistencyWithDeepLOB:
    """
    Tests that verify layout transformation produces correct DeepLOB input.
    
    DeepLOB paper (Section III.C):
    "xₜ = [p_a⁽ⁱ⁾(t), v_a⁽ⁱ⁾(t), p_b⁽ⁱ⁾(t), v_b⁽ⁱ⁾(t)]ᵢ₌₁ⁿ⁼¹⁰"
    
    The first conv layer (1×2 kernel, stride 1×2) pairs adjacent features:
    - Pairs (0,1): p_ask_L0 and v_ask_L0 → ask L0 info
    - Pairs (2,3): p_bid_L0 and v_bid_L0 → bid L0 info
    """
    
    def test_fi2010_adjacent_pairing(self):
        """
        Verify FI2010 layout pairs price/volume correctly for DeepLOB conv layer.
        
        A (1×2, stride 2) conv should pair:
        - features[0:2] = [p_ask_L0, v_ask_L0]
        - features[2:4] = [p_bid_L0, v_bid_L0]
        """
        num_levels = 10
        x = torch.zeros(1, 1, 40)
        
        # Set GROUPED layout values
        for level in range(num_levels):
            x[0, 0, level] = 100.0 + level * 0.01  # ask_prices (ascending)
            x[0, 0, 10 + level] = 100.0 + level  # ask_sizes
            x[0, 0, 20 + level] = 99.99 - level * 0.01  # bid_prices (descending)
            x[0, 0, 30 + level] = 50.0 + level  # bid_sizes
        
        y = rearrange_grouped_to_fi2010(x, num_levels)
        
        # Verify adjacent pairing for each level
        for level in range(num_levels):
            base = level * 4
            
            # Check p_ask[level] and v_ask[level] are adjacent
            p_ask = y[0, 0, base + 0].item()
            v_ask = y[0, 0, base + 1].item()
            
            # Check p_bid[level] and v_bid[level] are adjacent
            p_bid = y[0, 0, base + 2].item()
            v_bid = y[0, 0, base + 3].item()
            
            # Verify these match expected values from GROUPED
            # Use relaxed tolerance for f32 precision
            assert abs(p_ask - (100.0 + level * 0.01)) < 1e-4, f"Level {level} p_ask"
            assert abs(v_ask - (100.0 + level)) < 1e-4, f"Level {level} v_ask"
            assert abs(p_bid - (99.99 - level * 0.01)) < 1e-4, f"Level {level} p_bid"
            assert abs(v_bid - (50.0 + level)) < 1e-4, f"Level {level} v_bid"
