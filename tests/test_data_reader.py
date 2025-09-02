"""Tests for data reader."""

import os
import tempfile

import numpy as np
import pytest

from linear_regression.data.exceptions import InvalidCsvException
from linear_regression.data.reader import read_car_records


class TestReadCarRecords:
    """Tests for read_car_records function."""

    def test_read_valid_csv(self):
        """Test reading a valid CSV file."""
        csv_content = "km,price\n100000,5000\n200000,3000\n50000,8000"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_filename = f.name

        try:
            mileages, prices = read_car_records(temp_filename)

            expected_mileages = np.array([100000, 200000, 50000])
            expected_prices = np.array([5000, 3000, 8000])

            np.testing.assert_array_equal(mileages, expected_mileages)
            np.testing.assert_array_equal(prices, expected_prices)
        finally:
            os.unlink(temp_filename)

    def test_read_file_not_found(self):
        """Test reading a non-existent file."""
        with pytest.raises(InvalidCsvException, match="was not found"):
            read_car_records("non_existent_file.csv")

    def test_read_empty_file(self):
        """Test reading an empty CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_filename = f.name

        try:
            with pytest.raises(InvalidCsvException, match="is empty"):
                read_car_records(temp_filename)
        finally:
            os.unlink(temp_filename)

    def test_read_missing_columns(self):
        """Test reading CSV with missing required columns."""
        csv_content = "mileage,cost\n100000,5000\n200000,3000"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_filename = f.name

        try:
            with pytest.raises(InvalidCsvException, match="Missing required columns"):
                read_car_records(temp_filename)
        finally:
            os.unlink(temp_filename)

    def test_read_partial_missing_columns(self):
        """Test reading CSV with only one required column."""
        csv_content = "km,cost\n100000,5000\n200000,3000"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_filename = f.name

        try:
            with pytest.raises(InvalidCsvException, match="Missing required columns"):
                read_car_records(temp_filename)
        finally:
            os.unlink(temp_filename)

    def test_read_non_numeric_data(self):
        """Test reading CSV with non-numeric data (should convert to NaN)."""
        csv_content = "km,price\n100000,5000\nabc,3000\n50000,def"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_filename = f.name

        try:
            mileages, prices = read_car_records(temp_filename)

            # Should have NaN values where conversion failed
            assert len(mileages) == 3
            assert len(prices) == 3
            assert np.isnan(mileages[1])  # 'abc' -> NaN
            assert np.isnan(prices[2])  # 'def' -> NaN
            assert mileages[0] == 100000
            assert prices[0] == 5000
        finally:
            os.unlink(temp_filename)

    def test_read_extra_columns(self):
        """Test reading CSV with extra columns (should be ignored)."""
        csv_content = (
            "km,price,year,model\n100000,5000,2010,Toyota\n200000,3000,2005,Honda"
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_filename = f.name

        try:
            mileages, prices = read_car_records(temp_filename)

            expected_mileages = np.array([100000, 200000])
            expected_prices = np.array([5000, 3000])

            np.testing.assert_array_equal(mileages, expected_mileages)
            np.testing.assert_array_equal(prices, expected_prices)
        finally:
            os.unlink(temp_filename)

    def test_read_malformed_csv(self):
        """Test reading malformed CSV file."""
        csv_content = "km,price\n100000,5000,extra\n200000"  # Inconsistent columns

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_filename = f.name

        try:
            # pandas is usually forgiving with malformed CSV, but let's test
            mileages, prices = read_car_records(temp_filename)
            # Should handle gracefully, possibly with NaN values
            assert len(mileages) >= 1
            assert len(prices) >= 1
        finally:
            os.unlink(temp_filename)
