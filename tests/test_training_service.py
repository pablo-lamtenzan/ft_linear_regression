"""Tests for training service."""

import os
import tempfile

import pytest

from linear_regression.data.exceptions import InvalidCsvException
from linear_regression.model.linear_regression import LinearRegressionModel
from linear_regression.services.training import TrainingService


class TestTrainingService:
    """Tests for TrainingService class."""

    def create_test_data_file(self, data_content: str) -> str:
        """Helper to create temporary test data file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(data_content)
            return f.name

    def test_service_initialization_default(self):
        """Test service initialization with default parameters."""
        service = TrainingService()

        assert service.learning_rate == 0.01
        assert service.max_epochs == 1000
        assert service.tolerance == 1e-6
        assert service.model_save_path == "model.json"
        assert service.model is None

    def test_service_initialization_custom(self):
        """Test service initialization with custom parameters."""
        service = TrainingService(
            learning_rate=0.1,
            max_epochs=500,
            tolerance=1e-8,
            model_save_path="custom_model.json",
        )

        assert service.learning_rate == 0.1
        assert service.max_epochs == 500
        assert service.tolerance == 1e-8
        assert service.model_save_path == "custom_model.json"

    def test_train_model_valid_data(self):
        """Test training with valid data."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)
        model_file = "test_model.json"

        try:
            service = TrainingService(max_epochs=50, model_save_path=model_file)

            model = service.train_model(data_file, save_model=True, verbose=False)

            assert isinstance(model, LinearRegressionModel)
            assert model.is_trained
            assert service.model is not None
            assert service.model.is_trained
            assert os.path.exists(model_file)

        finally:
            os.unlink(data_file)
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_train_model_invalid_data(self):
        """Test training with invalid data file."""
        service = TrainingService()

        with pytest.raises(InvalidCsvException):
            service.train_model("non_existent_file.csv", verbose=False)

    def test_train_model_no_save(self):
        """Test training without saving model."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)
        model_file = "test_model.json"

        try:
            service = TrainingService(max_epochs=50, model_save_path=model_file)

            model = service.train_model(data_file, save_model=False, verbose=False)

            assert model.is_trained
            assert not os.path.exists(model_file)

        finally:
            os.unlink(data_file)

    def test_retrain_model(self):
        """Test retraining with different hyperparameters."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)

        try:
            service = TrainingService(max_epochs=10)

            # Initial training
            model1 = service.train_model(data_file, save_model=False, verbose=False)
            model1.get_training_metrics()

            # Retrain with different parameters
            model2 = service.retrain_model(
                data_file, learning_rate=0.1, max_epochs=20, verbose=False
            )
            metrics2 = model2.get_training_metrics()

            assert service.learning_rate == 0.1
            assert service.max_epochs == 20
            assert metrics2["epochs_trained"] <= 20

        finally:
            os.unlink(data_file)

    def test_evaluate_model_no_model(self):
        """Test evaluation without trained model."""
        service = TrainingService()

        with pytest.raises(InvalidCsvException, match="No trained model"):
            service.evaluate_model()

    def test_evaluate_model_trained(self):
        """Test evaluation with trained model."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)

        try:
            service = TrainingService(max_epochs=50)
            service.train_model(data_file, save_model=False, verbose=False)

            metrics = service.evaluate_model()

            assert "final_theta0" in metrics
            assert "final_theta1" in metrics
            assert "final_cost" in metrics
            assert "epochs_trained" in metrics
            assert "converged" in metrics

        finally:
            os.unlink(data_file)

    def test_evaluate_model_with_test_data(self):
        """Test evaluation with separate test data."""
        train_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        test_content = "km,price\n150000,7000\n250000,5000"

        train_file = self.create_test_data_file(train_content)
        test_file = self.create_test_data_file(test_content)

        try:
            service = TrainingService(max_epochs=50)
            service.train_model(train_file, save_model=False, verbose=False)

            metrics = service.evaluate_model(test_file)

            assert "test_mse" in metrics
            assert "test_samples" in metrics
            assert isinstance(metrics["test_mse"], float)
            assert isinstance(metrics["test_samples"], int)
            assert metrics["test_samples"] > 0

        finally:
            os.unlink(train_file)
            os.unlink(test_file)

    def test_get_model(self):
        """Test getting model instance."""
        service = TrainingService()

        # Initially None
        assert service.get_model() is None

        # After training
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)

        try:
            service.train_model(data_file, save_model=False, verbose=False)
            model = service.get_model()

            assert model is not None
            assert isinstance(model, LinearRegressionModel)
            assert model.is_trained

        finally:
            os.unlink(data_file)

    def test_load_existing_model(self):
        """Test loading existing model."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)
        model_file = "test_model.json"

        try:
            # Train and save model
            service1 = TrainingService(max_epochs=50, model_save_path=model_file)
            service1.train_model(data_file, save_model=True, verbose=False)

            # Load model in new service
            service2 = TrainingService()
            model = service2.load_existing_model(model_file)

            assert model.is_trained
            assert service2.model is not None
            assert service2.model.is_trained

            # Test prediction consistency
            pred1 = service1.model.predict(150000)
            pred2 = service2.model.predict(150000)
            assert abs(pred1 - pred2) < 1e-6

        finally:
            os.unlink(data_file)
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_load_existing_model_nonexistent(self):
        """Test loading non-existent model."""
        service = TrainingService()

        with pytest.raises(InvalidCsvException):
            service.load_existing_model("non_existent_model.json")

    def test_training_verbose_output(self, capsys):
        """Test verbose training output."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)

        try:
            service = TrainingService(max_epochs=10)
            service.train_model(data_file, save_model=False, verbose=True)

            captured = capsys.readouterr()
            assert "Starting training" in captured.out
            assert "Hyperparameters" in captured.out
            assert "Training completed" in captured.out
            assert "Final parameters" in captured.out

        finally:
            os.unlink(data_file)
