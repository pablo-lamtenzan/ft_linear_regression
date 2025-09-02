"""Data validation utilities for linear regression."""

import numpy as np

from .exceptions import InvalidCsvException


def validate_data(mileages: np.ndarray, prices: np.ndarray) -> None:
    """
    Validate mileage and price data for linear regression.

    Args:
        mileages: Array of mileage values
        prices: Array of price values

    Raises:
        InvalidCsvException: If data contains invalid values
    """
    # Check for NaN values
    if np.isnan(mileages).any() or np.isnan(prices).any():
        raise InvalidCsvException("Data contains missing or non-numeric values.")

    # Check for non-positive values
    if (mileages <= 0).any() or (prices <= 0).any():
        raise InvalidCsvException("Data contains non-positive values.")

    # Check for empty arrays
    if len(mileages) == 0 or len(prices) == 0:
        raise InvalidCsvException("Data arrays are empty.")

    # Check for mismatched lengths
    if len(mileages) != len(prices):
        raise InvalidCsvException(
            f"Mileage and price arrays have different lengths: "
            f"{len(mileages)} vs {len(prices)}"
        )


def validate_and_clean_data(
    mileages: np.ndarray, prices: np.ndarray, strict: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Validate and optionally clean data for linear regression.

    Args:
        mileages: Array of mileage values
        prices: Array of price values
        strict: If True, raise exceptions for invalid data.
                If False, remove invalid entries and continue.

    Returns:
        Tuple of (cleaned_mileages, cleaned_prices)

    Raises:
        InvalidCsvException: If strict=True and data contains invalid values
    """
    if strict:
        validate_data(mileages, prices)
        return mileages, prices

    # Non-strict mode: clean the data
    # Remove NaN values
    valid_mask = ~(np.isnan(mileages) | np.isnan(prices))
    mileages = mileages[valid_mask]
    prices = prices[valid_mask]

    # Remove non-positive values
    positive_mask = (mileages > 0) & (prices > 0)
    mileages = mileages[positive_mask]
    prices = prices[positive_mask]

    # Check if we have any data left
    if len(mileages) == 0:
        raise InvalidCsvException(
            "No valid data remaining after cleaning. "
            "All values were either missing, non-numeric, or non-positive."
        )

    return mileages, prices


def validate_prediction_input(mileage: float) -> None:
    """
    Validate input for price prediction.

    Args:
        mileage: Mileage value to validate

    Raises:
        InvalidCsvException: If mileage is invalid
    """
    if not isinstance(mileage, (int, float, np.integer, np.floating)):
        raise InvalidCsvException(
            f"Invalid mileage type: {type(mileage)}. Expected int or float."
        )

    if np.isnan(mileage) or np.isinf(mileage):
        raise InvalidCsvException("Mileage cannot be NaN or infinite.")

    if mileage <= 0:
        raise InvalidCsvException(
            f"Invalid mileage: {mileage}. Mileage must be greater than 0."
        )
