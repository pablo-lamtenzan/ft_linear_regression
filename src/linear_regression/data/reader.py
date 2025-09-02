"""CSV data reader for linear regression."""

import numpy as np
import pandas as pd

from .exceptions import InvalidCsvException


def read_car_records(filename: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read a CSV file and return mileage and price data.

    Args:
        filename: Path to the CSV file containing km,price data

    Returns:
        Tuple of (mileages, prices) as numpy arrays

    Raises:
        InvalidCsvException: If file not found or data format is invalid
    """
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError as e:
        raise InvalidCsvException(f"The file {filename} was not found.") from e
    except pd.errors.EmptyDataError as e:
        raise InvalidCsvException(f"The file {filename} is empty.") from e
    except pd.errors.ParserError as e:
        raise InvalidCsvException(f"Error parsing CSV file {filename}: {e}") from e

    # Check if required columns exist
    required_columns = ["km", "price"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise InvalidCsvException(
            f"Missing required columns: {missing_columns}. "
            f"Expected columns: {required_columns}"
        )

    # Convert columns to numeric, coercing errors to NaN
    try:
        df["km"] = pd.to_numeric(df["km"], errors="coerce")
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    except Exception as e:
        raise InvalidCsvException(f"Error converting data to numeric: {e}") from e

    # Extract the data as numpy arrays
    mileages = df["km"].to_numpy()
    prices = df["price"].to_numpy()

    return mileages, prices
