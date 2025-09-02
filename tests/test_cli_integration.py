"""End-to-end integration tests for CLI programs."""

import json
import os
import tempfile

from click.testing import CliRunner

from linear_regression.cli.predict import main as predict_main
from linear_regression.cli.train import main as train_main


class TestCLIIntegration:
    """Integration tests for CLI programs."""

    def create_test_data_file(self, data_content: str) -> str:
        """Helper to create temporary test data file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(data_content)
            return f.name

    def test_train_basic_functionality(self):
        """Test basic training functionality."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000\n150000,7000"
        data_file = self.create_test_data_file(data_content)
        model_file = "test_model.json"

        try:
            runner = CliRunner()

            # Test training
            result = runner.invoke(
                train_main,
                [
                    data_file,
                    "--max-epochs",
                    "50",
                    "--model-file",
                    model_file,
                    "--quiet",
                ],
            )

            assert result.exit_code == 0
            assert os.path.exists(model_file)

            # Verify model file content
            with open(model_file) as f:
                model_data = json.load(f)
                assert "theta0" in model_data
                assert "theta1" in model_data
                assert "scaler" in model_data

        finally:
            os.unlink(data_file)
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_train_with_invalid_data(self):
        """Test training with invalid data file."""
        runner = CliRunner()

        result = runner.invoke(train_main, ["non_existent_file.csv"])

        assert result.exit_code != 0  # Click returns 2 for file not found
        assert "does not exist" in result.output or "not found" in result.output

    def test_train_with_invalid_parameters(self):
        """Test training with invalid parameters."""
        data_content = "km,price\n100000,8000\n200000,6000"
        data_file = self.create_test_data_file(data_content)

        try:
            runner = CliRunner()

            # Test negative learning rate
            result = runner.invoke(train_main, [data_file, "--learning-rate", "-0.01"])

            assert result.exit_code == 1
            assert "Learning rate must be positive" in result.output

            # Test zero max epochs
            result = runner.invoke(train_main, [data_file, "--max-epochs", "0"])

            assert result.exit_code == 1
            assert "Max epochs must be positive" in result.output

        finally:
            os.unlink(data_file)

    def test_predict_basic_functionality(self):
        """Test basic prediction functionality."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000\n150000,7000"
        data_file = self.create_test_data_file(data_content)
        model_file = "test_model.json"

        try:
            runner = CliRunner()

            # First train a model
            train_result = runner.invoke(
                train_main,
                [
                    data_file,
                    "--max-epochs",
                    "50",
                    "--model-file",
                    model_file,
                    "--quiet",
                ],
            )
            assert train_result.exit_code == 0

            # Test prediction
            predict_result = runner.invoke(
                predict_main, ["150000", "--model-file", model_file]
            )

            assert predict_result.exit_code == 0
            # Should output a numeric value
            output = predict_result.output.strip()
            price = float(output)
            assert price > 0

        finally:
            os.unlink(data_file)
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_predict_verbose_output(self):
        """Test prediction with verbose output."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000\n150000,7000"
        data_file = self.create_test_data_file(data_content)
        model_file = "test_model.json"

        try:
            runner = CliRunner()

            # Train model
            train_result = runner.invoke(
                train_main,
                [
                    data_file,
                    "--max-epochs",
                    "50",
                    "--model-file",
                    model_file,
                    "--quiet",
                ],
            )
            assert train_result.exit_code == 0

            # Test verbose prediction
            predict_result = runner.invoke(
                predict_main, ["150000", "--model-file", model_file, "--verbose"]
            )

            assert predict_result.exit_code == 0
            assert "Model loaded successfully" in predict_result.output
            assert "Parameters:" in predict_result.output
            assert "Predicted price:" in predict_result.output

        finally:
            os.unlink(data_file)
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_predict_without_model(self):
        """Test prediction without trained model."""
        runner = CliRunner()

        result = runner.invoke(
            predict_main, ["150000", "--model-file", "non_existent_model.json"]
        )

        assert result.exit_code == 1
        assert "Model file not found" in result.output

    def test_predict_invalid_mileage(self):
        """Test prediction with invalid mileage."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)
        model_file = "test_model.json"

        try:
            runner = CliRunner()

            # Train model
            train_result = runner.invoke(
                train_main,
                [
                    data_file,
                    "--max-epochs",
                    "50",
                    "--model-file",
                    model_file,
                    "--quiet",
                ],
            )
            assert train_result.exit_code == 0

            # Test zero mileage (invalid)
            predict_result = runner.invoke(
                predict_main, ["0", "--model-file", model_file]
            )

            assert predict_result.exit_code == 1
            assert "Input error" in predict_result.output

        finally:
            os.unlink(data_file)
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_predict_range_checking(self):
        """Test prediction with range checking."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)
        model_file = "test_model.json"

        try:
            runner = CliRunner()

            # Train model
            train_result = runner.invoke(
                train_main,
                [
                    data_file,
                    "--max-epochs",
                    "50",
                    "--model-file",
                    model_file,
                    "--quiet",
                ],
            )
            assert train_result.exit_code == 0

            # Test prediction outside range with range checking
            predict_result = runner.invoke(
                predict_main,
                [
                    "500000",  # Outside training range
                    "--model-file",
                    model_file,
                    "--check-range",
                ],
            )

            assert predict_result.exit_code == 0
            assert "Warning:" in predict_result.output

        finally:
            os.unlink(data_file)
            if os.path.exists(model_file):
                os.unlink(model_file)

    def test_train_no_save_option(self):
        """Test training without saving model."""
        data_content = "km,price\n100000,8000\n200000,6000\n300000,4000"
        data_file = self.create_test_data_file(data_content)
        model_file = "test_model.json"

        try:
            runner = CliRunner()

            result = runner.invoke(
                train_main,
                [
                    data_file,
                    "--max-epochs",
                    "50",
                    "--model-file",
                    model_file,
                    "--no-save",
                    "--quiet",
                ],
            )

            assert result.exit_code == 0
            assert not os.path.exists(model_file)

        finally:
            os.unlink(data_file)

    def test_complete_workflow(self):
        """Test complete training and prediction workflow."""
        data_content = (
            "km,price\n100000,8000\n200000,6000\n300000,4000\n150000,7000\n250000,5000"
        )
        data_file = self.create_test_data_file(data_content)
        model_file = "test_model.json"

        try:
            runner = CliRunner()

            # Step 1: Train model
            train_result = runner.invoke(
                train_main,
                [
                    data_file,
                    "--learning-rate",
                    "0.01",
                    "--max-epochs",
                    "100",
                    "--model-file",
                    model_file,
                    "--quiet",
                ],
            )

            assert train_result.exit_code == 0
            assert os.path.exists(model_file)

            # Step 2: Make predictions
            test_mileages = ["120000", "180000", "220000"]

            for mileage in test_mileages:
                predict_result = runner.invoke(
                    predict_main, [mileage, "--model-file", model_file]
                )

                assert predict_result.exit_code == 0

                # Verify output is a valid price
                price = float(predict_result.output.strip())
                assert price > 0
                assert price < 20000  # Reasonable upper bound

            # Step 3: Test verbose prediction
            verbose_result = runner.invoke(
                predict_main, ["150000", "--model-file", model_file, "--verbose"]
            )

            assert verbose_result.exit_code == 0
            assert "Model loaded successfully" in verbose_result.output
            assert "θ₀=" in verbose_result.output
            assert "θ₁=" in verbose_result.output

        finally:
            os.unlink(data_file)
            if os.path.exists(model_file):
                os.unlink(model_file)
