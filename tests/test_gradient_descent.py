"""Tests for gradient descent algorithm."""

import pytest

from linear_regression.math.gradient_descent import (
    gradient_descent_step,
    train_with_gradient_descent,
)


class TestGradientDescentStep:
    """Tests for gradient_descent_step function."""

    def test_gradient_descent_step_basic(self):
        """Test basic gradient descent step."""
        theta0, theta1 = 0.0, 0.0
        mileages = [0.0, 0.5, 1.0]
        prices = [0.0, 0.5, 1.0]  # Perfect linear relationship
        learning_rate = 0.1

        new_theta0, new_theta1 = gradient_descent_step(
            theta0, theta1, mileages, prices, learning_rate
        )

        # With perfect linear data and zero initial parameters,
        # the algorithm should move towards the correct parameters
        assert new_theta0 != theta0 or new_theta1 != theta1

    def test_gradient_descent_step_no_change_perfect_fit(self):
        """Test gradient descent step with perfect parameters."""
        # For y = x, perfect parameters are theta0=0, theta1=1
        theta0, theta1 = 0.0, 1.0
        mileages = [0.0, 0.5, 1.0]
        prices = [0.0, 0.5, 1.0]
        learning_rate = 0.1

        new_theta0, new_theta1 = gradient_descent_step(
            theta0, theta1, mileages, prices, learning_rate
        )

        # Parameters should remain approximately the same
        assert abs(new_theta0 - theta0) < 1e-10
        assert abs(new_theta1 - theta1) < 1e-10

    def test_gradient_descent_step_single_point(self):
        """Test gradient descent step with single data point."""
        theta0, theta1 = 0.0, 0.0
        mileages = [1.0]
        prices = [2.0]
        learning_rate = 0.1

        new_theta0, new_theta1 = gradient_descent_step(
            theta0, theta1, mileages, prices, learning_rate
        )

        # Should update parameters based on the single point
        assert new_theta0 != theta0
        assert new_theta1 != theta1


class TestTrainWithGradientDescent:
    """Tests for train_with_gradient_descent function."""

    def test_train_simple_linear_data(self):
        """Test training with simple linear data."""
        # Perfect linear relationship: y = 2x + 1
        mileages = [0.0, 0.25, 0.5, 0.75, 1.0]
        prices = [1.0, 1.5, 2.0, 2.5, 3.0]

        theta0, theta1, theta0_hist, theta1_hist, cost_hist = (
            train_with_gradient_descent(
                mileages, prices, learning_rate=0.1, max_epochs=1000, tolerance=1e-8
            )
        )

        # Should converge close to the true parameters (theta0=1, theta1=2)
        assert abs(theta0 - 1.0) < 0.1
        assert abs(theta1 - 2.0) < 0.1

        # History should be recorded
        assert len(theta0_hist) > 1
        assert len(theta1_hist) > 1
        assert len(cost_hist) > 0

        # Cost should decrease over time
        assert cost_hist[-1] < cost_hist[0]

    def test_train_convergence(self):
        """Test that training converges."""
        mileages = [0.0, 0.5, 1.0]
        prices = [0.0, 0.5, 1.0]

        theta0, theta1, _, _, cost_hist = train_with_gradient_descent(
            mileages, prices, learning_rate=0.5, max_epochs=100, tolerance=1e-6
        )

        # Should converge before max epochs with higher learning rate
        assert len(cost_hist) < 100

    def test_train_max_epochs_reached(self):
        """Test training when max epochs is reached."""
        mileages = [0.0, 0.5, 1.0]
        prices = [0.0, 0.5, 1.0]

        theta0, theta1, _, _, cost_hist = train_with_gradient_descent(
            mileages, prices, learning_rate=0.001, max_epochs=5, tolerance=1e-10
        )

        # Should run for exactly max_epochs
        assert len(cost_hist) == 5

    def test_train_empty_data_raises_error(self):
        """Test that empty data raises an error."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            train_with_gradient_descent([], [], learning_rate=0.1)

    def test_train_single_point(self):
        """Test training with single data point."""
        mileages = [0.5]
        prices = [1.0]

        theta0, theta1, _, _, cost_hist = train_with_gradient_descent(
            mileages, prices, learning_rate=0.1, max_epochs=10
        )

        # Should complete without error
        assert isinstance(theta0, float)
        assert isinstance(theta1, float)
        assert len(cost_hist) > 0

    def test_train_different_learning_rates(self):
        """Test training with different learning rates."""
        mileages = [0.0, 0.5, 1.0]
        prices = [0.0, 0.5, 1.0]

        # High learning rate
        _, _, _, _, cost_hist_high = train_with_gradient_descent(
            mileages, prices, learning_rate=0.5, max_epochs=50
        )

        # Low learning rate
        _, _, _, _, cost_hist_low = train_with_gradient_descent(
            mileages, prices, learning_rate=0.01, max_epochs=50
        )

        # Both should work but may converge at different rates
        assert len(cost_hist_high) > 0
        assert len(cost_hist_low) > 0
