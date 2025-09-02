"""Tests for data validator."""

import numpy as np
import pytest

from linear_regression.data.exceptions import InvalidCsvException
from linear_regression.data.validator import (
    validate_and_clean_data,
    validate_data,
    validate_prediction_input,
)


class TestValidateData:
    """Tests for validate_data function."""

    def test_validate_valid_data(self):
        """Test validation with valid data."""
        mileages = np.array([100000, 200000, 50000])
        prices = np.array([5000, 3000, 8000])

        # Should not raise any exception
        validate_data(mileages, prices)

    def test_validate_nan_mileages(self):
        """Test validation with NaN mileages."""
        mileages = np.array([100000, np.nan, 50000])
        prices = np.array([5000, 3000, 8000])

        with pytest.raises(InvalidCsvException, match="missing or non-numeric values"):
            validate_data(mileages, prices)

    def test_validate_nan_prices(self):
        """Test validation with NaN prices."""
        mileages = np.array([100000, 200000, 50000])
        prices = np.array([5000, np.nan, 8000])

        with pytest.raises(InvalidCsvException, match="missing or non-numeric values"):
            validate_data(mileages, prices)

    def test_validate_negative_mileages(self):
        """Test validation with negative mileages."""
        mileages = np.array([100000, -200000, 50000])
        prices = np.array([5000, 3000, 8000])

        with pytest.raises(InvalidCsvException, match="non-positive values"):
            validate_data(mileages, prices)

    def test_validate_zero_mileages(self):
        """Test validation with zero mileages."""
        mileages = np.array([100000, 0, 50000])
        prices = np.array([5000, 3000, 8000])

        with pytest.raises(InvalidCsvException, match="non-positive values"):
            validate_data(mileages, prices)

    def test_validate_negative_prices(self):
        """Test validation with negative prices."""
        mileages = np.array([100000, 200000, 50000])
        prices = np.array([5000, -3000, 8000])

        with pytest.raises(InvalidCsvException, match="non-positive values"):
            validate_data(mileages, prices)

    def test_validate_empty_arrays(self):
        """Test validation with empty arrays."""
        mileages = np.array([])
        prices = np.array([])

        with pytest.raises(InvalidCsvException, match="empty"):
            validate_data(mileages, prices)

    def test_validate_mismatched_lengths(self):
        """Test validation with mismatched array lengths."""
        mileages = np.array([100000, 200000])
        prices = np.array([5000, 3000, 8000])

        with pytest.raises(InvalidCsvException, match="different lengths"):
            validate_data(mileages, prices)


class TestValidateAndCleanData:
    """Tests for validate_and_clean_data function."""

    def test_validate_clean_strict_mode(self):
        """Test strict mode validation."""
        mileages = np.array([100000, 200000, 50000])
        prices = np.array([5000, 3000, 8000])

        clean_mileages, clean_prices = validate_and_clean_data(
            mileages, prices, strict=True
        )

        np.testing.assert_array_equal(clean_mileages, mileages)
        np.testing.assert_array_equal(clean_prices, prices)

    def test_validate_clean_strict_mode_invalid_data(self):
        """Test strict mode with invalid data."""
        mileages = np.array([100000, np.nan, 50000])
        prices = np.array([5000, 3000, 8000])

        with pytest.raises(InvalidCsvException):
            validate_and_clean_data(mileages, prices, strict=True)

    def test_validate_clean_non_strict_mode(self):
        """Test non-strict mode with mixed valid/invalid data."""
        mileages = np.array([100000, np.nan, 50000, -10000, 0])
        prices = np.array([5000, 3000, np.nan, 8000, 1000])

        clean_mileages, clean_prices = validate_and_clean_data(
            mileages, prices, strict=False
        )

        # Should keep only the first entry (100000, 5000)
        expected_mileages = np.array([100000])
        expected_prices = np.array([5000])

        np.testing.assert_array_equal(clean_mileages, expected_mileages)
        np.testing.assert_array_equal(clean_prices, expected_prices)

    def test_validate_clean_non_strict_all_invalid(self):
        """Test non-strict mode with all invalid data."""
        mileages = np.array([np.nan, -100, 0])
        prices = np.array([np.nan, -500, 0])

        with pytest.raises(InvalidCsvException, match="No valid data remaining"):
            validate_and_clean_data(mileages, prices, strict=False)


class TestValidatePredictionInput:
    """Tests for validate_prediction_input function."""

    def test_validate_valid_float(self):
        """Test validation with valid float."""
        validate_prediction_input(100000.5)  # Should not raise

    def test_validate_valid_int(self):
        """Test validation with valid int."""
        validate_prediction_input(100000)  # Should not raise

    def test_validate_invalid_type(self):
        """Test validation with invalid type."""
        with pytest.raises(InvalidCsvException, match="Invalid mileage type"):
            validate_prediction_input("100000")

    def test_validate_nan_input(self):
        """Test validation with NaN input."""
        with pytest.raises(InvalidCsvException, match="cannot be NaN"):
            validate_prediction_input(float("nan"))

    def test_validate_infinite_input(self):
        """Test validation with infinite input."""
        with pytest.raises(InvalidCsvException, match="cannot be NaN or infinite"):
            validate_prediction_input(float("inf"))

    def test_validate_negative_input(self):
        """Test validation with negative input."""
        with pytest.raises(InvalidCsvException, match="must be greater than 0"):
            validate_prediction_input(-100000)

    def test_validate_zero_input(self):
        """Test validation with zero input."""
        with pytest.raises(InvalidCsvException, match="must be greater than 0"):
            validate_prediction_input(0)
