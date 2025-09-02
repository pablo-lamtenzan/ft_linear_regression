"""Precision calculation service for linear regression model."""

from typing import Any

import numpy as np

from ..data.exceptions import InvalidCsvException
from ..data.reader import read_car_records
from ..data.validator import validate_and_clean_data
from ..model.linear_regression import LinearRegressionModel


class PrecisionService:
    """Service for calculating precision metrics of linear regression model."""

    def __init__(self) -> None:
        """Initialize the precision service."""
        pass

    def calculate_r_squared(
        self, data_file: str, model: LinearRegressionModel
    ) -> float:
        """
        Calculate R-squared (coefficient of determination) for the model.

        R² = 1 - (SS_res / SS_tot)
        where:
        - SS_res = Σ(y_actual - y_predicted)²
        - SS_tot = Σ(y_actual - y_mean)²

        Args:
            data_file: Path to CSV file containing test data
            model: Trained linear regression model

        Returns:
            R-squared value (0 to 1, where 1 is perfect fit)

        Raises:
            InvalidCsvException: If model is not trained or data cannot be loaded
        """
        if not model.is_trained:
            raise InvalidCsvException("Model is not trained.")

        # Load data
        try:
            mileages, prices = read_car_records(data_file)
            mileages, prices = validate_and_clean_data(mileages, prices, strict=False)
        except Exception as e:
            raise InvalidCsvException(f"Failed to load data: {e}") from e

        # Make predictions
        predictions = []
        actuals = []

        for mileage, actual_price in zip(mileages, prices):
            try:
                pred = model.predict(mileage)
                predictions.append(pred)
                actuals.append(actual_price)
            except Exception:
                continue

        if len(predictions) < 2:
            raise InvalidCsvException(
                "Insufficient valid data points for R² calculation."
            )

        # Calculate R-squared
        actuals = np.array(actuals)
        predictions = np.array(predictions)

        # Sum of squares of residuals
        ss_res: float = np.sum((actuals - predictions) ** 2)

        # Total sum of squares
        y_mean = np.mean(actuals)
        ss_tot: float = np.sum((actuals - y_mean) ** 2)

        # R-squared
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0

        r_squared = 1 - (ss_res / ss_tot)
        return float(r_squared)

    def calculate_mean_absolute_error(
        self, data_file: str, model: LinearRegressionModel
    ) -> float:
        """
        Calculate Mean Absolute Error (MAE) for the model.

        MAE = (1/n) * Σ|y_actual - y_predicted|

        Args:
            data_file: Path to CSV file containing test data
            model: Trained linear regression model

        Returns:
            Mean absolute error value

        Raises:
            InvalidCsvException: If model is not trained or data cannot be loaded
        """
        if not model.is_trained:
            raise InvalidCsvException("Model is not trained.")

        # Load data
        try:
            mileages, prices = read_car_records(data_file)
            mileages, prices = validate_and_clean_data(mileages, prices, strict=False)
        except Exception as e:
            raise InvalidCsvException(f"Failed to load data: {e}") from e

        # Calculate absolute errors
        absolute_errors = []

        for mileage, actual_price in zip(mileages, prices):
            try:
                pred = model.predict(mileage)
                absolute_errors.append(abs(actual_price - pred))
            except Exception:
                continue

        if not absolute_errors:
            raise InvalidCsvException("No valid predictions could be made.")

        return float(np.mean(absolute_errors))

    def calculate_root_mean_squared_error(
        self, data_file: str, model: LinearRegressionModel
    ) -> float:
        """
        Calculate Root Mean Squared Error (RMSE) for the model.

        RMSE = √[(1/n) * Σ(y_actual - y_predicted)²]

        Args:
            data_file: Path to CSV file containing test data
            model: Trained linear regression model

        Returns:
            Root mean squared error value

        Raises:
            InvalidCsvException: If model is not trained or data cannot be loaded
        """
        if not model.is_trained:
            raise InvalidCsvException("Model is not trained.")

        # Load data
        try:
            mileages, prices = read_car_records(data_file)
            mileages, prices = validate_and_clean_data(mileages, prices, strict=False)
        except Exception as e:
            raise InvalidCsvException(f"Failed to load data: {e}") from e

        # Calculate squared errors
        squared_errors = []

        for mileage, actual_price in zip(mileages, prices):
            try:
                pred = model.predict(mileage)
                squared_errors.append((actual_price - pred) ** 2)
            except Exception:
                continue

        if not squared_errors:
            raise InvalidCsvException("No valid predictions could be made.")

        mse = np.mean(squared_errors)
        return float(np.sqrt(mse))

    def calculate_mean_absolute_percentage_error(
        self, data_file: str, model: LinearRegressionModel
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE) for the model.

        MAPE = (100/n) * Σ|((y_actual - y_predicted) / y_actual)|

        Args:
            data_file: Path to CSV file containing test data
            model: Trained linear regression model

        Returns:
            Mean absolute percentage error (as percentage)

        Raises:
            InvalidCsvException: If model is not trained or data cannot be loaded
        """
        if not model.is_trained:
            raise InvalidCsvException("Model is not trained.")

        # Load data
        try:
            mileages, prices = read_car_records(data_file)
            mileages, prices = validate_and_clean_data(mileages, prices, strict=False)
        except Exception as e:
            raise InvalidCsvException(f"Failed to load data: {e}") from e

        # Calculate percentage errors
        percentage_errors = []

        for mileage, actual_price in zip(mileages, prices):
            try:
                if actual_price == 0:
                    continue  # Skip zero values to avoid division by zero

                pred = model.predict(mileage)
                percentage_error = abs((actual_price - pred) / actual_price)
                percentage_errors.append(percentage_error)
            except Exception:
                continue

        if not percentage_errors:
            raise InvalidCsvException("No valid predictions could be made.")

        return float(np.mean(percentage_errors) * 100)

    def calculate_comprehensive_metrics(
        self, data_file: str, model: LinearRegressionModel
    ) -> dict[str, Any]:
        """
        Calculate comprehensive precision metrics for the model.

        Args:
            data_file: Path to CSV file containing test data
            model: Trained linear regression model

        Returns:
            Dictionary containing all precision metrics

        Raises:
            InvalidCsvException: If model is not trained or data cannot be loaded
        """
        metrics: dict[str, Any] = {}

        try:
            metrics["r_squared"] = self.calculate_r_squared(data_file, model)
        except Exception as e:
            metrics["r_squared"] = None
            metrics["r_squared_error"] = str(e)

        try:
            metrics["mae"] = self.calculate_mean_absolute_error(data_file, model)
        except Exception as e:
            metrics["mae"] = None
            metrics["mae_error"] = str(e)

        try:
            metrics["rmse"] = self.calculate_root_mean_squared_error(data_file, model)
        except Exception as e:
            metrics["rmse"] = None
            metrics["rmse_error"] = str(e)

        try:
            metrics["mape"] = self.calculate_mean_absolute_percentage_error(
                data_file, model
            )
        except Exception as e:
            metrics["mape"] = None
            metrics["mape_error"] = str(e)

        return metrics

    def generate_precision_report(
        self, data_file: str, model: LinearRegressionModel, verbose: bool = True
    ) -> str:
        """
        Generate a comprehensive precision report.

        Args:
            data_file: Path to CSV file containing test data
            model: Trained linear regression model
            verbose: Whether to print the report

        Returns:
            Formatted precision report as string

        Raises:
            InvalidCsvException: If model is not trained or data cannot be loaded
        """
        metrics = self.calculate_comprehensive_metrics(data_file, model)

        report_lines = [
            "=" * 50,
            "LINEAR REGRESSION MODEL PRECISION REPORT",
            "=" * 50,
            "",
            f"Data file: {data_file}",
            f"Model trained: {'Yes' if model.is_trained else 'No'}",
            "",
            "PRECISION METRICS:",
            "-" * 20,
        ]

        # R-squared
        if metrics.get("r_squared") is not None:
            r_sq = metrics["r_squared"]
            report_lines.extend(
                [
                    f"R² (Coefficient of Determination): {r_sq:.6f}",
                    f"  → Explains {r_sq * 100:.2f}% of variance in the data",
                    f"  → Quality: {'Excellent' if r_sq > 0.9 else 'Good' if r_sq > 0.7 else 'Fair' if r_sq > 0.5 else 'Poor'}",
                    "",
                ]
            )
        else:
            report_lines.extend(
                [
                    "R² (Coefficient of Determination): ERROR",
                    f"  → {metrics.get('r_squared_error', 'Unknown error')}",
                    "",
                ]
            )

        # MAE
        if metrics.get("mae") is not None:
            report_lines.extend(
                [
                    f"MAE (Mean Absolute Error): ${metrics['mae']:.2f}",
                    "  → Average prediction error magnitude",
                    "",
                ]
            )

        # RMSE
        if metrics.get("rmse") is not None:
            report_lines.extend(
                [
                    f"RMSE (Root Mean Squared Error): ${metrics['rmse']:.2f}",
                    "  → Penalizes larger errors more heavily",
                    "",
                ]
            )

        # MAPE
        if metrics.get("mape") is not None:
            report_lines.extend(
                [
                    f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%",
                    "  → Average percentage error",
                    "",
                ]
            )

        report_lines.extend(["=" * 50, ""])

        report = "\n".join(report_lines)

        if verbose:
            print(report)

        return report
