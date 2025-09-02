"""Tests for mathematical utilities."""

import pytest

from linear_regression.math.utils import (
    calculate_mse,
    denormalize,
    linear_prediction,
    normalize,
)


class TestNormalize:
    """Tests for normalize function."""

    def test_normalize_basic(self):
        """Test basic normalization."""
        result = normalize(5.0, 0.0, 10.0)
        assert result == 0.5

    def test_normalize_min_value(self):
        """Test normalization of minimum value."""
        result = normalize(0.0, 0.0, 10.0)
        assert result == 0.0

    def test_normalize_max_value(self):
        """Test normalization of maximum value."""
        result = normalize(10.0, 0.0, 10.0)
        assert result == 1.0

    def test_normalize_negative_range(self):
        """Test normalization with negative values."""
        result = normalize(-5.0, -10.0, 0.0)
        assert result == 0.5

    def test_normalize_equal_min_max_raises_error(self):
        """Test that equal min and max values raise ValueError."""
        with pytest.raises(ValueError, match="Maximum and minimum values are equal"):
            normalize(5.0, 5.0, 5.0)


class TestDenormalize:
    """Tests for denormalize function."""

    def test_denormalize_basic(self):
        """Test basic denormalization."""
        result = denormalize(0.5, 0.0, 10.0)
        assert result == 5.0

    def test_denormalize_min_value(self):
        """Test denormalization of 0."""
        result = denormalize(0.0, 0.0, 10.0)
        assert result == 0.0

    def test_denormalize_max_value(self):
        """Test denormalization of 1."""
        result = denormalize(1.0, 0.0, 10.0)
        assert result == 10.0

    def test_denormalize_negative_range(self):
        """Test denormalization with negative values."""
        result = denormalize(0.5, -10.0, 0.0)
        assert result == -5.0

    def test_normalize_denormalize_roundtrip(self):
        """Test that normalize and denormalize are inverse operations."""
        original = 7.5
        min_val, max_val = 0.0, 10.0

        normalized = normalize(original, min_val, max_val)
        denormalized = denormalize(normalized, min_val, max_val)

        assert abs(denormalized - original) < 1e-10


class TestCalculateMse:
    """Tests for calculate_mse function."""

    def test_calculate_mse_perfect_predictions(self):
        """Test MSE with perfect predictions."""
        actuals = [1.0, 2.0, 3.0, 4.0]
        predictions = [1.0, 2.0, 3.0, 4.0]
        result = calculate_mse(actuals, predictions)
        assert result == 0.0

    def test_calculate_mse_basic(self):
        """Test MSE calculation with simple values."""
        actuals = [1.0, 2.0, 3.0]
        predictions = [1.5, 2.5, 3.5]
        # Squared errors: [0.25, 0.25, 0.25]
        # MSE = (0.25 + 0.25 + 0.25) / (2 * 3) = 0.75 / 6 = 0.125
        result = calculate_mse(actuals, predictions)
        assert abs(result - 0.125) < 1e-10

    def test_calculate_mse_single_value(self):
        """Test MSE with single value."""
        actuals = [5.0]
        predictions = [3.0]
        # Squared error: (5-3)^2 = 4
        # MSE = 4 / (2 * 1) = 2.0
        result = calculate_mse(actuals, predictions)
        assert result == 2.0

    def test_calculate_mse_different_lengths_raises_error(self):
        """Test that different length lists raise ValueError."""
        actuals = [1.0, 2.0, 3.0]
        predictions = [1.0, 2.0]

        with pytest.raises(
            ValueError, match="length of actuals and predictions must be the same"
        ):
            calculate_mse(actuals, predictions)

    def test_calculate_mse_empty_lists_raises_error(self):
        """Test that empty lists raise ValueError."""
        with pytest.raises(ValueError):
            calculate_mse([], [])


class TestLinearPrediction:
    """Tests for linear_prediction function."""

    def test_linear_prediction_basic(self):
        """Test basic linear prediction."""
        result = linear_prediction(slope=2.0, x=3.0, y_intercept=1.0)
        # y = 2 * 3 + 1 = 7
        assert result == 7.0

    def test_linear_prediction_zero_slope(self):
        """Test linear prediction with zero slope."""
        result = linear_prediction(slope=0.0, x=5.0, y_intercept=3.0)
        assert result == 3.0

    def test_linear_prediction_zero_intercept(self):
        """Test linear prediction with zero intercept."""
        result = linear_prediction(slope=2.0, x=4.0, y_intercept=0.0)
        assert result == 8.0

    def test_linear_prediction_negative_values(self):
        """Test linear prediction with negative values."""
        result = linear_prediction(slope=-1.5, x=2.0, y_intercept=-3.0)
        # y = -1.5 * 2 + (-3) = -3 - 3 = -6
        assert result == -6.0

    def test_linear_prediction_zero_x(self):
        """Test linear prediction with x=0."""
        result = linear_prediction(slope=5.0, x=0.0, y_intercept=10.0)
        assert result == 10.0
