"""Tests for data preprocessor."""

import numpy as np
import pytest

from linear_regression.data.exceptions import InvalidCsvException
from linear_regression.data.preprocessor import DataPreprocessor, DataScaler


class TestDataPreprocessor:
    """Tests for DataPreprocessor class."""

    def test_fit_transform_basic(self):
        """Test basic fit_transform functionality."""
        preprocessor = DataPreprocessor()
        mileages = np.array([100000, 200000, 300000])
        prices = np.array([8000, 6000, 4000])

        norm_mileages, norm_prices, scaler = preprocessor.fit_transform(
            mileages, prices
        )

        # Check normalization
        assert norm_mileages[0] == 0.0  # min value
        assert norm_mileages[-1] == 1.0  # max value
        assert norm_mileages[1] == 0.5  # middle value

        assert norm_prices[0] == 1.0  # max price (8000)
        assert norm_prices[-1] == 0.0  # min price (4000)
        assert norm_prices[1] == 0.5  # middle price (6000)

        # Check scaler
        assert scaler.km_min == 100000
        assert scaler.km_max == 300000
        assert scaler.price_min == 4000
        assert scaler.price_max == 8000

    def test_fit_transform_equal_mileages(self):
        """Test fit_transform with equal mileage values."""
        preprocessor = DataPreprocessor()
        mileages = np.array([100000, 100000, 100000])
        prices = np.array([8000, 6000, 4000])

        with pytest.raises(
            InvalidCsvException, match="All mileage values are the same"
        ):
            preprocessor.fit_transform(mileages, prices)

    def test_fit_transform_equal_prices(self):
        """Test fit_transform with equal price values."""
        preprocessor = DataPreprocessor()
        mileages = np.array([100000, 200000, 300000])
        prices = np.array([5000, 5000, 5000])

        with pytest.raises(InvalidCsvException, match="All price values are the same"):
            preprocessor.fit_transform(mileages, prices)

    def test_transform_mileage_fitted(self):
        """Test transform_mileage with fitted preprocessor."""
        preprocessor = DataPreprocessor()
        mileages = np.array([100000, 200000, 300000])
        prices = np.array([8000, 6000, 4000])

        preprocessor.fit_transform(mileages, prices)

        # Test transformation of new mileage
        normalized = preprocessor.transform_mileage(150000)
        expected = (150000 - 100000) / (300000 - 100000)  # 0.25
        assert abs(normalized - expected) < 1e-10

    def test_transform_mileage_not_fitted(self):
        """Test transform_mileage without fitting first."""
        preprocessor = DataPreprocessor()

        with pytest.raises(InvalidCsvException, match="not fitted"):
            preprocessor.transform_mileage(150000)

    def test_inverse_transform_price_fitted(self):
        """Test inverse_transform_price with fitted preprocessor."""
        preprocessor = DataPreprocessor()
        mileages = np.array([100000, 200000, 300000])
        prices = np.array([4000, 6000, 8000])

        preprocessor.fit_transform(mileages, prices)

        # Test inverse transformation
        original_price = preprocessor.inverse_transform_price(0.5)
        expected = 0.5 * (8000 - 4000) + 4000  # 6000
        assert abs(original_price - expected) < 1e-10

    def test_inverse_transform_price_not_fitted(self):
        """Test inverse_transform_price without fitting first."""
        preprocessor = DataPreprocessor()

        with pytest.raises(InvalidCsvException, match="not fitted"):
            preprocessor.inverse_transform_price(0.5)

    def test_inverse_transform_data_fitted(self):
        """Test inverse_transform_data with fitted preprocessor."""
        preprocessor = DataPreprocessor()
        mileages = np.array([100000, 200000, 300000])
        prices = np.array([4000, 6000, 8000])

        norm_mileages, norm_prices, _ = preprocessor.fit_transform(mileages, prices)

        # Transform back
        orig_mileages, orig_prices = preprocessor.inverse_transform_data(
            norm_mileages, norm_prices
        )

        # Should get back original values
        np.testing.assert_array_almost_equal(orig_mileages, mileages)
        np.testing.assert_array_almost_equal(orig_prices, prices)

    def test_inverse_transform_data_not_fitted(self):
        """Test inverse_transform_data without fitting first."""
        preprocessor = DataPreprocessor()
        norm_mileages = np.array([0.0, 0.5, 1.0])
        norm_prices = np.array([0.0, 0.5, 1.0])

        with pytest.raises(InvalidCsvException, match="not fitted"):
            preprocessor.inverse_transform_data(norm_mileages, norm_prices)

    def test_scaler_property(self):
        """Test scaler property."""
        preprocessor = DataPreprocessor()

        # Initially None
        assert preprocessor.scaler is None

        # After fitting
        mileages = np.array([100000, 200000, 300000])
        prices = np.array([4000, 6000, 8000])
        preprocessor.fit_transform(mileages, prices)

        scaler = preprocessor.scaler
        assert scaler is not None
        assert scaler.km_min == 100000
        assert scaler.km_max == 300000
        assert scaler.price_min == 4000
        assert scaler.price_max == 8000

    def test_roundtrip_transformation(self):
        """Test that fit_transform and inverse_transform are inverse operations."""
        preprocessor = DataPreprocessor()
        original_mileages = np.array([50000, 100000, 150000, 200000, 250000])
        original_prices = np.array([10000, 8000, 6000, 4000, 2000])

        # Forward transformation
        norm_mileages, norm_prices, _ = preprocessor.fit_transform(
            original_mileages, original_prices
        )

        # Backward transformation
        recovered_mileages, recovered_prices = preprocessor.inverse_transform_data(
            norm_mileages, norm_prices
        )

        # Should recover original values
        np.testing.assert_array_almost_equal(recovered_mileages, original_mileages)
        np.testing.assert_array_almost_equal(recovered_prices, original_prices)


class TestDataScaler:
    """Tests for DataScaler NamedTuple."""

    def test_data_scaler_creation(self):
        """Test DataScaler creation and access."""
        scaler = DataScaler(
            km_min=100000, km_max=300000, price_min=4000, price_max=8000
        )

        assert scaler.km_min == 100000
        assert scaler.km_max == 300000
        assert scaler.price_min == 4000
        assert scaler.price_max == 8000

    def test_data_scaler_immutable(self):
        """Test that DataScaler is immutable."""
        scaler = DataScaler(
            km_min=100000, km_max=300000, price_min=4000, price_max=8000
        )

        # Should not be able to modify
        with pytest.raises(AttributeError):
            scaler.km_min = 200000
