"""Prediction service for linear regression model."""

import os
from typing import Optional

from ..data.exceptions import InvalidArgException, InvalidCsvException
from ..model.linear_regression import LinearRegressionModel


class PredictionService:
    """Service for making predictions with trained linear regression models."""

    def __init__(self, model_file: str = "model.json") -> None:
        """
        Initialize the prediction service.

        Args:
            model_file: Path to saved model file
        """
        self.model_file = model_file
        self.model: Optional[LinearRegressionModel] = None
        self._model_loaded = False

    def load_model(self, model_file: Optional[str] = None) -> None:
        """
        Load a trained model for predictions.

        Args:
            model_file: Optional path to model file (uses default if not provided)

        Raises:
            InvalidCsvException: If model cannot be loaded
        """
        file_path = model_file or self.model_file

        if not os.path.exists(file_path):
            raise InvalidCsvException(
                f"Model file not found: {file_path}. "
                "Please train a model first using the training program."
            )

        self.model = LinearRegressionModel()
        self.model.load_model(file_path)
        self._model_loaded = True

    def predict_single(self, mileage: float, verbose: bool = False) -> float:
        """
        Predict car price for a single mileage value.

        Args:
            mileage: Car mileage in kilometers
            verbose: Whether to print prediction details

        Returns:
            Predicted price

        Raises:
            InvalidCsvException: If model is not loaded
            InvalidArgException: If mileage is invalid
        """
        if not self._model_loaded or self.model is None:
            raise InvalidCsvException("No model loaded. Call load_model() first.")

        try:
            # Validate input
            if not isinstance(mileage, (int, float)):
                raise InvalidArgException(
                    f"Invalid mileage type: {type(mileage)}. Expected number."
                )

            if mileage <= 0:
                raise InvalidArgException(
                    f"Invalid mileage: {mileage}. Must be greater than 0."
                )

            # Make prediction
            prediction = self.model.predict(mileage)

            if verbose:
                print(f"Input mileage: {mileage:,.0f} km")
                print(f"Predicted price: ${prediction:,.2f}")

            return prediction

        except Exception as e:
            if isinstance(e, (InvalidCsvException, InvalidArgException)):
                raise
            else:
                raise InvalidCsvException(f"Prediction failed: {e}") from e

    def predict_batch(
        self, mileages: list[float], verbose: bool = False
    ) -> list[tuple[float, float]]:
        """
        Predict car prices for multiple mileage values.

        Args:
            mileages: List of car mileages in kilometers
            verbose: Whether to print prediction details

        Returns:
            List of (mileage, predicted_price) tuples

        Raises:
            InvalidCsvException: If model is not loaded
            InvalidArgException: If any mileage is invalid
        """
        if not self._model_loaded or self.model is None:
            raise InvalidCsvException("No model loaded. Call load_model() first.")

        if not mileages:
            raise InvalidArgException("Empty mileage list provided.")

        results = []

        for i, mileage in enumerate(mileages):
            try:
                prediction = self.predict_single(mileage, verbose=False)
                results.append((mileage, prediction))

                if verbose:
                    print(f"{i + 1:2d}. {mileage:8,.0f} km -> ${prediction:8,.2f}")

            except Exception as e:
                if verbose:
                    print(f"{i + 1:2d}. {mileage:8,.0f} km -> ERROR: {e}")
                # Skip invalid entries but continue processing
                continue

        if not results:
            raise InvalidArgException(
                "No valid predictions could be made from the provided mileages."
            )

        return results

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information

        Raises:
            InvalidCsvException: If model is not loaded
        """
        if not self._model_loaded or self.model is None:
            raise InvalidCsvException("No model loaded. Call load_model() first.")

        return {
            "model_file": self.model_file,
            "is_trained": self.model.is_trained,
            "theta0": self.model.theta0,
            "theta1": self.model.theta1,
            "scaler": {
                "km_min": self.model.scaler.km_min,
                "km_max": self.model.scaler.km_max,
                "price_min": self.model.scaler.price_min,
                "price_max": self.model.scaler.price_max,
            }
            if self.model.scaler
            else None,
            "hyperparameters": {
                "learning_rate": self.model.learning_rate,
                "max_epochs": self.model.max_epochs,
                "tolerance": self.model.tolerance,
            },
        }

    def validate_prediction_range(self, mileage: float) -> dict:
        """
        Validate if mileage is within reasonable range of training data.

        Args:
            mileage: Mileage to validate

        Returns:
            Dictionary with validation results

        Raises:
            InvalidCsvException: If model is not loaded
        """
        if not self._model_loaded or self.model is None:
            raise InvalidCsvException("No model loaded. Call load_model() first.")

        if not self.model.scaler:
            return {"within_range": True, "warning": None, "training_range": None}

        km_min = self.model.scaler.km_min
        km_max = self.model.scaler.km_max

        # Check if within training range
        within_range = km_min <= mileage <= km_max

        # Generate warnings for extrapolation
        warning = None
        if mileage < km_min:
            warning = (
                f"Mileage {mileage:,.0f} is below training range "
                f"({km_min:,.0f} - {km_max:,.0f}). "
                "Prediction may be unreliable."
            )
        elif mileage > km_max:
            warning = (
                f"Mileage {mileage:,.0f} is above training range "
                f"({km_min:,.0f} - {km_max:,.0f}). "
                "Prediction may be unreliable."
            )

        return {
            "within_range": within_range,
            "warning": warning,
            "training_range": (km_min, km_max),
        }

    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is loaded."""
        return self._model_loaded
