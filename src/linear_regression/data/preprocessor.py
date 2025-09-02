"""Data preprocessing utilities for linear regression."""

from typing import NamedTuple, Union

import numpy as np

from ..math.utils import denormalize, normalize
from .exceptions import InvalidCsvException


class DataScaler(NamedTuple):
    """Container for data scaling parameters."""

    km_min: float
    km_max: float
    price_min: float
    price_max: float


class DataPreprocessor:
    """Data preprocessor for linear regression with normalization capabilities."""

    def __init__(self) -> None:
        """Initialize the data preprocessor."""
        self._scaler: Union[DataScaler, None] = None

    def fit_transform(
        self, mileages: np.ndarray, prices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, DataScaler]:
        """
        Fit the scaler to the data and transform it.

        Args:
            mileages: Array of mileage values
            prices: Array of price values

        Returns:
            Tuple of (normalized_mileages, normalized_prices, scaler)

        Raises:
            InvalidCsvException: If data cannot be normalized
        """
        # Calculate min/max values for normalization
        km_min, km_max = float(np.min(mileages)), float(np.max(mileages))
        price_min, price_max = float(np.min(prices)), float(np.max(prices))

        # Check for equal min/max (cannot normalize)
        if km_min == km_max:
            raise InvalidCsvException(
                f"All mileage values are the same ({km_min}). Cannot normalize."
            )

        if price_min == price_max:
            raise InvalidCsvException(
                f"All price values are the same ({price_min}). Cannot normalize."
            )

        # Create scaler
        scaler = DataScaler(
            km_min=km_min, km_max=km_max, price_min=price_min, price_max=price_max
        )

        # Normalize the data
        normalized_mileages = np.array([normalize(m, km_min, km_max) for m in mileages])
        normalized_prices = np.array(
            [normalize(p, price_min, price_max) for p in prices]
        )

        self._scaler = scaler
        return normalized_mileages, normalized_prices, scaler

    def transform_mileage(self, mileage: float) -> float:
        """
        Transform a single mileage value using the fitted scaler.

        Args:
            mileage: Mileage value to normalize

        Returns:
            Normalized mileage value

        Raises:
            InvalidCsvException: If scaler is not fitted
        """
        if self._scaler is None:
            raise InvalidCsvException(
                "Preprocessor not fitted. Call fit_transform first."
            )

        return normalize(mileage, self._scaler.km_min, self._scaler.km_max)

    def inverse_transform_price(self, normalized_price: float) -> float:
        """
        Transform a normalized price back to original scale.

        Args:
            normalized_price: Normalized price value

        Returns:
            Price in original scale

        Raises:
            InvalidCsvException: If scaler is not fitted
        """
        if self._scaler is None:
            raise InvalidCsvException(
                "Preprocessor not fitted. Call fit_transform first."
            )

        return denormalize(
            normalized_price, self._scaler.price_min, self._scaler.price_max
        )

    def inverse_transform_data(
        self, normalized_mileages: np.ndarray, normalized_prices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Transform normalized data back to original scale.

        Args:
            normalized_mileages: Array of normalized mileage values
            normalized_prices: Array of normalized price values

        Returns:
            Tuple of (original_mileages, original_prices)

        Raises:
            InvalidCsvException: If scaler is not fitted
        """
        if self._scaler is None:
            raise InvalidCsvException(
                "Preprocessor not fitted. Call fit_transform first."
            )

        original_mileages = np.array(
            [
                denormalize(m, self._scaler.km_min, self._scaler.km_max)
                for m in normalized_mileages
            ]
        )
        original_prices = np.array(
            [
                denormalize(p, self._scaler.price_min, self._scaler.price_max)
                for p in normalized_prices
            ]
        )

        return original_mileages, original_prices

    @property
    def scaler(self) -> Union[DataScaler, None]:
        """Get the fitted scaler parameters."""
        return self._scaler
