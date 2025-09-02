"""Linear regression model implementation."""

import json
from typing import Union

from ..data.exceptions import InvalidCsvException
from ..data.preprocessor import DataPreprocessor, DataScaler
from ..data.reader import read_car_records
from ..data.validator import validate_and_clean_data, validate_prediction_input
from ..math.gradient_descent import train_with_gradient_descent
from ..math.utils import linear_prediction


class LinearRegressionModel:
    """
    Linear regression model for car price prediction based on mileage.

    This model uses gradient descent to learn the relationship between
    car mileage and price using the equation: price = theta0 + theta1 * mileage
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        tolerance: float = 1e-6,
    ) -> None:
        """
        Initialize the linear regression model.

        Args:
            learning_rate: Learning rate for gradient descent
            max_epochs: Maximum number of training epochs
            tolerance: Convergence tolerance for early stopping
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance

        # Model parameters
        self.theta0: float = 0.0  # y-intercept
        self.theta1: float = 0.0  # slope

        # Training history
        self.theta0_history: list[float] = []
        self.theta1_history: list[float] = []
        self.cost_history: list[float] = []

        # Data preprocessing
        self.preprocessor = DataPreprocessor()
        self.scaler: Union[DataScaler, None] = None
        self._is_trained = False

    def load_data(self, filename: str, validate: bool = True) -> None:
        """
        Load training data from CSV file.

        Args:
            filename: Path to CSV file containing km,price data
            validate: Whether to validate the data strictly

        Raises:
            InvalidCsvException: If data is invalid or cannot be loaded
        """
        # Read raw data
        mileages, prices = read_car_records(filename)

        # Validate and clean data
        self.mileages, self.prices = validate_and_clean_data(
            mileages, prices, strict=validate
        )

        # Preprocess data (normalize)
        self.normalized_mileages, self.normalized_prices, self.scaler = (
            self.preprocessor.fit_transform(self.mileages, self.prices)
        )

    def train(self) -> None:
        """
        Train the linear regression model using gradient descent.

        Raises:
            InvalidCsvException: If no data has been loaded
        """
        if not hasattr(self, "normalized_mileages"):
            raise InvalidCsvException("No data loaded. Call load_data() first.")

        # Train using gradient descent
        (
            self.theta0,
            self.theta1,
            self.theta0_history,
            self.theta1_history,
            self.cost_history,
        ) = train_with_gradient_descent(
            self.normalized_mileages.tolist(),
            self.normalized_prices.tolist(),
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            tolerance=self.tolerance,
        )

        self._is_trained = True

    def predict(self, mileage: float) -> float:
        """
        Predict car price for given mileage.

        Args:
            mileage: Car mileage in kilometers

        Returns:
            Predicted price

        Raises:
            InvalidCsvException: If model is not trained or input is invalid
        """
        if not self._is_trained:
            raise InvalidCsvException("Model not trained. Call train() first.")

        # Validate input
        validate_prediction_input(mileage)

        # Normalize input mileage
        normalized_mileage = self.preprocessor.transform_mileage(mileage)

        # Make prediction in normalized space
        normalized_prediction = linear_prediction(
            slope=self.theta1, x=normalized_mileage, y_intercept=self.theta0
        )

        # Denormalize prediction
        price = self.preprocessor.inverse_transform_price(normalized_prediction)

        return price

    def get_training_metrics(self) -> dict:
        """
        Get training metrics and history.

        Returns:
            Dictionary containing training metrics
        """
        if not self._is_trained:
            return {}

        return {
            "final_theta0": self.theta0,
            "final_theta1": self.theta1,
            "final_cost": self.cost_history[-1] if self.cost_history else None,
            "epochs_trained": len(self.cost_history),
            "converged": len(self.cost_history) < self.max_epochs,
            "theta0_history": self.theta0_history.copy(),
            "theta1_history": self.theta1_history.copy(),
            "cost_history": self.cost_history.copy(),
        }

    def save_model(self, filename: str) -> None:
        """
        Save trained model parameters to file.

        Args:
            filename: Path to save model parameters

        Raises:
            InvalidCsvException: If model is not trained
        """
        if not self._is_trained or self.scaler is None:
            raise InvalidCsvException("Model not trained. Call train() first.")

        model_data = {
            "theta0": self.theta0,
            "theta1": self.theta1,
            "scaler": {
                "km_min": self.scaler.km_min,
                "km_max": self.scaler.km_max,
                "price_min": self.scaler.price_min,
                "price_max": self.scaler.price_max,
            },
            "hyperparameters": {
                "learning_rate": self.learning_rate,
                "max_epochs": self.max_epochs,
                "tolerance": self.tolerance,
            },
        }

        with open(filename, "w") as f:
            json.dump(model_data, f, indent=2)

    def load_model(self, filename: str) -> None:
        """
        Load trained model parameters from file.

        Args:
            filename: Path to load model parameters from

        Raises:
            InvalidCsvException: If file cannot be loaded or is invalid
        """
        try:
            with open(filename) as f:
                model_data = json.load(f)
        except FileNotFoundError as e:
            raise InvalidCsvException(f"Model file {filename} not found.") from e
        except json.JSONDecodeError as e:
            raise InvalidCsvException(f"Invalid model file format: {e}") from e

        try:
            # Load parameters
            self.theta0 = model_data["theta0"]
            self.theta1 = model_data["theta1"]

            # Recreate scaler
            scaler_data = model_data["scaler"]
            self.scaler = DataScaler(
                km_min=scaler_data["km_min"],
                km_max=scaler_data["km_max"],
                price_min=scaler_data["price_min"],
                price_max=scaler_data["price_max"],
            )

            # Set scaler in preprocessor
            self.preprocessor._scaler = self.scaler

            # Load hyperparameters
            hyperparams = model_data["hyperparameters"]
            self.learning_rate = hyperparams["learning_rate"]
            self.max_epochs = hyperparams["max_epochs"]
            self.tolerance = hyperparams["tolerance"]

            self._is_trained = True

        except KeyError as e:
            raise InvalidCsvException(
                f"Missing required field in model file: {e}"
            ) from e

    @property
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained
