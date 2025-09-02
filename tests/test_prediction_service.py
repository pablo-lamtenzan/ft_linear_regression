"""Tests for prediction service."""

import os
import tempfile

import pytest

from linear_regression.data.exceptions import InvalidArgException, InvalidCsvException
from linear_regression.model.linear_regression import LinearRegressionModel
from linear_regression.services.prediction import PredictionService


class TestPredictionService:
    """Tests for PredictionService class."""

    def create_test_data_file(self, data_content: str) -> str:
        """Helper to create temporary test data file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(data_content)
            return f.name

    def create_test_model_file(self) -> str:
        """Helper to create a test model file."""
        # Create and train a simple model
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)
        model_file = "test_model.json"

        try:
            model = LinearRegressionModel(max_epochs=50)
            model.load_data(data_file)
            model.train()
            model.save_model(model_file)
            return model_file
        finally:
            os.unlink(data_file)

    def test_service_initialization(self):
        """Test service initialization."""
        service = PredictionService()
        assert service.model_file == "model.json"
        assert service.model is None
        assert not service.is_model_loaded

        service_custom = PredictionService("custom_model.json")
        assert service_custom.model_file == "custom_model.json"

    def test_load_model_nonexistent(self):
        """Test loading non-existent model."""
        service = PredictionService()

        with pytest.raises(InvalidCsvException, match="Model file not found"):
            service.load_model("non_existent_model.json")

    def test_load_model_valid(self):
        """Test loading valid model."""
        model_file = self.create_test_model_file()

        try:
            service = PredictionService()
            service.load_model(model_file)

            assert service.is_model_loaded
            assert service.model is not None
            assert service.model.is_trained
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_predict_single_no_model(self):
        """Test prediction without loaded model."""
        service = PredictionService()

        with pytest.raises(InvalidCsvException, match="No model loaded"):
            service.predict_single(150000)

    def test_predict_single_invalid_input(self):
        """Test prediction with invalid input."""
        model_file = self.create_test_model_file()

        try:
            service = PredictionService()
            service.load_model(model_file)

            # Test invalid type
            with pytest.raises(InvalidArgException, match="Invalid mileage type"):
                service.predict_single("150000")

            # Test negative value
            with pytest.raises(InvalidArgException, match="Must be greater than 0"):
                service.predict_single(-150000)

            # Test zero value
            with pytest.raises(InvalidArgException, match="Must be greater than 0"):
                service.predict_single(0)
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_predict_single_valid(self):
        """Test valid single prediction."""
        model_file = self.create_test_model_file()

        try:
            service = PredictionService()
            service.load_model(model_file)

            prediction = service.predict_single(150000)

            assert isinstance(prediction, float)
            assert prediction > 0
            # For the test data, should be reasonable
            assert 1000 < prediction < 10000
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_predict_batch_no_model(self):
        """Test batch prediction without loaded model."""
        service = PredictionService()

        with pytest.raises(InvalidCsvException, match="No model loaded"):
            service.predict_batch([150000, 250000])

    def test_predict_batch_empty_list(self):
        """Test batch prediction with empty list."""
        model_file = self.create_test_model_file()

        try:
            service = PredictionService()
            service.load_model(model_file)

            with pytest.raises(InvalidArgException, match="Empty mileage list"):
                service.predict_batch([])
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_predict_batch_valid(self):
        """Test valid batch prediction."""
        model_file = self.create_test_model_file()

        try:
            service = PredictionService()
            service.load_model(model_file)

            mileages = [150000, 250000]
            results = service.predict_batch(mileages)

            assert len(results) == 2
            for mileage, prediction in results:
                assert mileage in mileages
                assert isinstance(prediction, float)
                assert prediction > 0
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_predict_batch_mixed_valid_invalid(self):
        """Test batch prediction with mixed valid/invalid inputs."""
        model_file = self.create_test_model_file()

        try:
            service = PredictionService()
            service.load_model(model_file)

            # Mix of valid and invalid mileages
            mileages = [150000, -100000, 250000, 0]
            results = service.predict_batch(mileages, verbose=False)

            # Should only return results for valid mileages
            assert len(results) == 2
            valid_mileages = [r[0] for r in results]
            assert 150000 in valid_mileages
            assert 250000 in valid_mileages
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_get_model_info_no_model(self):
        """Test getting model info without loaded model."""
        service = PredictionService()

        with pytest.raises(InvalidCsvException, match="No model loaded"):
            service.get_model_info()

    def test_get_model_info_valid(self):
        """Test getting model info with loaded model."""
        model_file = self.create_test_model_file()

        try:
            service = PredictionService()
            service.load_model(model_file)

            info = service.get_model_info()

            assert "model_file" in info
            assert "is_trained" in info
            assert "theta0" in info
            assert "theta1" in info
            assert "scaler" in info
            assert "hyperparameters" in info

            assert info["is_trained"] is True
            assert isinstance(info["theta0"], float)
            assert isinstance(info["theta1"], float)
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_validate_prediction_range_no_model(self):
        """Test range validation without loaded model."""
        service = PredictionService()

        with pytest.raises(InvalidCsvException, match="No model loaded"):
            service.validate_prediction_range(150000)

    def test_validate_prediction_range_within(self):
        """Test range validation for mileage within training range."""
        model_file = self.create_test_model_file()

        try:
            service = PredictionService()
            service.load_model(model_file)

            # Test mileage within training range (100000-300000)
            validation = service.validate_prediction_range(200000)

            assert validation["within_range"] is True
            assert validation["warning"] is None
            assert validation["training_range"] == (100000, 300000)
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_validate_prediction_range_below(self):
        """Test range validation for mileage below training range."""
        model_file = self.create_test_model_file()

        try:
            service = PredictionService()
            service.load_model(model_file)

            # Test mileage below training range
            validation = service.validate_prediction_range(50000)

            assert validation["within_range"] is False
            assert "below training range" in validation["warning"]
            assert validation["training_range"] == (100000, 300000)
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_validate_prediction_range_above(self):
        """Test range validation for mileage above training range."""
        model_file = self.create_test_model_file()

        try:
            service = PredictionService()
            service.load_model(model_file)

            # Test mileage above training range
            validation = service.validate_prediction_range(400000)

            assert validation["within_range"] is False
            assert "above training range" in validation["warning"]
            assert validation["training_range"] == (100000, 300000)
        finally:
            if os.path.exists(model_file):
                os.unlink(model_file)
