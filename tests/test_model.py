"""Tests for linear regression model."""

import os
import tempfile

import pytest

from linear_regression.data.exceptions import InvalidCsvException
from linear_regression.model.linear_regression import LinearRegressionModel


class TestLinearRegressionModel:
    """Tests for LinearRegressionModel class."""

    def create_test_data_file(self, data_content: str) -> str:
        """Helper to create temporary test data file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(data_content)
            return f.name

    def test_model_initialization(self):
        """Test model initialization with default parameters."""
        model = LinearRegressionModel()

        assert model.learning_rate == 0.01
        assert model.max_epochs == 1000
        assert model.tolerance == 1e-6
        assert model.theta0 == 0.0
        assert model.theta1 == 0.0
        assert not model.is_trained

    def test_model_initialization_custom_params(self):
        """Test model initialization with custom parameters."""
        model = LinearRegressionModel(learning_rate=0.1, max_epochs=500, tolerance=1e-8)

        assert model.learning_rate == 0.1
        assert model.max_epochs == 500
        assert model.tolerance == 1e-8

    def test_load_data_valid(self):
        """Test loading valid data."""
        data_content = "km,price\n100000,5000\n200000,3000\n50000,8000"
        data_file = self.create_test_data_file(data_content)

        try:
            model = LinearRegressionModel()
            model.load_data(data_file)

            assert hasattr(model, "mileages")
            assert hasattr(model, "prices")
            assert hasattr(model, "normalized_mileages")
            assert hasattr(model, "normalized_prices")
            assert hasattr(model, "scaler")

            assert len(model.mileages) == 3
            assert len(model.prices) == 3
        finally:
            os.unlink(data_file)

    def test_load_data_invalid_file(self):
        """Test loading data from non-existent file."""
        model = LinearRegressionModel()

        with pytest.raises(InvalidCsvException, match="not found"):
            model.load_data("non_existent_file.csv")

    def test_train_without_data(self):
        """Test training without loading data first."""
        model = LinearRegressionModel()

        with pytest.raises(InvalidCsvException, match="No data loaded"):
            model.train()

    def test_train_with_data(self):
        """Test training with valid data."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)

        try:
            model = LinearRegressionModel(max_epochs=100)
            model.load_data(data_file)
            model.train()

            assert model.is_trained
            assert len(model.theta0_history) > 0
            assert len(model.theta1_history) > 0
            assert len(model.cost_history) > 0
        finally:
            os.unlink(data_file)

    def test_predict_without_training(self):
        """Test prediction without training first."""
        model = LinearRegressionModel()

        with pytest.raises(InvalidCsvException, match="not trained"):
            model.predict(150000)

    def test_predict_with_invalid_input(self):
        """Test prediction with invalid input."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)

        try:
            model = LinearRegressionModel(max_epochs=10)
            model.load_data(data_file)
            model.train()

            with pytest.raises(InvalidCsvException, match="must be greater than 0"):
                model.predict(-100000)

            with pytest.raises(InvalidCsvException, match="must be greater than 0"):
                model.predict(0)
        finally:
            os.unlink(data_file)

    def test_predict_valid(self):
        """Test valid prediction."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)

        try:
            model = LinearRegressionModel(max_epochs=100)
            model.load_data(data_file)
            model.train()

            prediction = model.predict(150000)

            # Should return a reasonable price
            assert isinstance(prediction, float)
            assert prediction > 0
            # For this linear data, prediction should be around 7000
            assert 5000 < prediction < 9000
        finally:
            os.unlink(data_file)

    def test_get_training_metrics_untrained(self):
        """Test getting metrics from untrained model."""
        model = LinearRegressionModel()
        metrics = model.get_training_metrics()

        assert metrics == {}

    def test_get_training_metrics_trained(self):
        """Test getting metrics from trained model."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)

        try:
            model = LinearRegressionModel(max_epochs=50)
            model.load_data(data_file)
            model.train()

            metrics = model.get_training_metrics()

            assert "final_theta0" in metrics
            assert "final_theta1" in metrics
            assert "final_cost" in metrics
            assert "epochs_trained" in metrics
            assert "converged" in metrics
            assert "theta0_history" in metrics
            assert "theta1_history" in metrics
            assert "cost_history" in metrics

            assert isinstance(metrics["final_theta0"], float)
            assert isinstance(metrics["final_theta1"], float)
            assert isinstance(metrics["final_cost"], float)
            assert isinstance(metrics["epochs_trained"], int)
            assert isinstance(metrics["converged"], bool)
        finally:
            os.unlink(data_file)

    def test_save_model_untrained(self):
        """Test saving untrained model."""
        model = LinearRegressionModel()

        with pytest.raises(InvalidCsvException, match="not trained"):
            model.save_model("test_model.json")

    def test_save_load_model(self):
        """Test saving and loading trained model."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)
        model_file = "test_model.json"

        try:
            # Train and save model
            model1 = LinearRegressionModel(max_epochs=50)
            model1.load_data(data_file)
            model1.train()
            model1.save_model(model_file)

            # Load model
            model2 = LinearRegressionModel()
            model2.load_model(model_file)

            # Compare parameters
            assert abs(model1.theta0 - model2.theta0) < 1e-10
            assert abs(model1.theta1 - model2.theta1) < 1e-10
            assert model2.is_trained

            # Test prediction consistency
            test_mileage = 150000
            pred1 = model1.predict(test_mileage)
            pred2 = model2.predict(test_mileage)
            assert abs(pred1 - pred2) < 1e-6

        finally:
            os.unlink(data_file)
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_load_model_nonexistent(self):
        """Test loading non-existent model file."""
        model = LinearRegressionModel()

        with pytest.raises(InvalidCsvException, match="not found"):
            model.load_model("non_existent_model.json")

    def test_load_model_invalid_format(self):
        """Test loading model with invalid JSON format."""
        model_file = "invalid_model.json"

        try:
            with open(model_file, "w") as f:
                f.write("invalid json content")

            model = LinearRegressionModel()
            with pytest.raises(InvalidCsvException, match="Invalid model file format"):
                model.load_model(model_file)
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)
