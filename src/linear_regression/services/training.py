"""Training service for linear regression model."""

import os
from typing import Optional

from ..data.exceptions import InvalidCsvException
from ..model.linear_regression import LinearRegressionModel


class TrainingService:
    """Service for training linear regression models."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        tolerance: float = 1e-6,
        model_save_path: str = "model.json",
    ) -> None:
        """
        Initialize the training service.

        Args:
            learning_rate: Learning rate for gradient descent
            max_epochs: Maximum number of training epochs
            tolerance: Convergence tolerance for early stopping
            model_save_path: Path to save trained model
        """
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.model_save_path = model_save_path

        self.model: Optional[LinearRegressionModel] = None

    def train_model(
        self,
        data_file: str,
        validate_data: bool = True,
        save_model: bool = True,
        verbose: bool = True,
    ) -> LinearRegressionModel:
        """
        Train a linear regression model on the provided data.

        Args:
            data_file: Path to CSV file containing training data
            validate_data: Whether to validate data strictly
            save_model: Whether to save the trained model
            verbose: Whether to print training progress

        Returns:
            Trained LinearRegressionModel instance

        Raises:
            InvalidCsvException: If training fails
        """
        if verbose:
            print(f"Starting training with data from: {data_file}")
            print("Hyperparameters:")
            print(f"  Learning rate: {self.learning_rate}")
            print(f"  Max epochs: {self.max_epochs}")
            print(f"  Tolerance: {self.tolerance}")

        # Create and configure model
        self.model = LinearRegressionModel(
            learning_rate=self.learning_rate,
            max_epochs=self.max_epochs,
            tolerance=self.tolerance,
        )

        try:
            # Load and preprocess data
            if verbose:
                print("Loading and preprocessing data...")
            self.model.load_data(data_file, validate=validate_data)

            # Train the model
            if verbose:
                print("Training model...")
            self.model.train()

            # Get training metrics
            metrics = self.model.get_training_metrics()

            if verbose:
                print("Training completed!")
                print("Final parameters:")
                print(f"  θ₀ (intercept): {metrics['final_theta0']:.6f}")
                print(f"  θ₁ (slope): {metrics['final_theta1']:.6f}")
                print(f"  Final cost: {metrics['final_cost']:.6f}")
                print(f"  Epochs trained: {metrics['epochs_trained']}")
                print(f"  Converged: {metrics['converged']}")

            # Save model if requested
            if save_model:
                self.model.save_model(self.model_save_path)
                if verbose:
                    print(f"Model saved to: {self.model_save_path}")

            return self.model

        except Exception as e:
            raise InvalidCsvException(f"Training failed: {e}") from e

    def retrain_model(
        self,
        data_file: str,
        learning_rate: Optional[float] = None,
        max_epochs: Optional[int] = None,
        tolerance: Optional[float] = None,
        verbose: bool = True,
    ) -> LinearRegressionModel:
        """
        Retrain model with different hyperparameters.

        Args:
            data_file: Path to CSV file containing training data
            learning_rate: New learning rate (optional)
            max_epochs: New max epochs (optional)
            tolerance: New tolerance (optional)
            verbose: Whether to print training progress

        Returns:
            Retrained LinearRegressionModel instance
        """
        # Update hyperparameters if provided
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if max_epochs is not None:
            self.max_epochs = max_epochs
        if tolerance is not None:
            self.tolerance = tolerance

        if verbose:
            print("Retraining model with updated hyperparameters...")

        return self.train_model(data_file, verbose=verbose)

    def evaluate_model(self, test_data_file: Optional[str] = None) -> dict:
        """
        Evaluate the trained model.

        Args:
            test_data_file: Optional test data file for evaluation

        Returns:
            Dictionary containing evaluation metrics

        Raises:
            InvalidCsvException: If model is not trained
        """
        if self.model is None or not self.model.is_trained:
            raise InvalidCsvException("No trained model available for evaluation.")

        metrics = self.model.get_training_metrics()

        # If test data provided, evaluate on it
        if test_data_file and os.path.exists(test_data_file):
            # Load test data
            test_model = LinearRegressionModel()
            test_model.load_data(test_data_file, validate=False)

            # Make predictions on test data
            test_predictions = []
            for mileage in test_model.mileages:
                try:
                    pred = self.model.predict(float(mileage))
                    test_predictions.append(pred)
                except Exception:
                    continue

            if test_predictions:
                # Calculate test MSE
                test_errors = [
                    (actual - pred) ** 2
                    for actual, pred in zip(
                        test_model.prices[: len(test_predictions)], test_predictions
                    )
                ]
                test_mse = sum(test_errors) / (2 * len(test_errors))
                metrics["test_mse"] = test_mse
                metrics["test_samples"] = len(test_predictions)

        return metrics

    def get_model(self) -> Optional[LinearRegressionModel]:
        """Get the trained model instance."""
        return self.model

    def load_existing_model(self, model_file: str) -> LinearRegressionModel:
        """
        Load an existing trained model.

        Args:
            model_file: Path to saved model file

        Returns:
            Loaded LinearRegressionModel instance
        """
        self.model = LinearRegressionModel()
        self.model.load_model(model_file)
        return self.model
