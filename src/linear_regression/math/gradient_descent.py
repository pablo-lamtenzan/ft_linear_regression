"""Gradient descent algorithm implementation."""

from .utils import linear_prediction


def gradient_descent_step(
    theta0: float,
    theta1: float,
    mileages: list[float],
    prices: list[float],
    learning_rate: float,
) -> tuple[float, float]:
    """
    Perform one step of gradient descent.

    Following the formulas from the subject:
    tmpθ₀ = learningRate * (1/m) * Σ(estimatePrice(mileage[i]) - price[i])
    tmpθ₁ = learningRate * (1/m) * Σ((estimatePrice(mileage[i]) - price[i]) * mileage[i])

    Args:
        theta0: Current y-intercept parameter
        theta1: Current slope parameter
        mileages: List of mileage values (normalized)
        prices: List of price values (normalized)
        learning_rate: Learning rate for gradient descent

    Returns:
        Tuple of (new_theta0, new_theta1)
    """
    m = len(mileages)

    # Calculate predictions for all data points
    predictions = [linear_prediction(theta1, mileage, theta0) for mileage in mileages]

    # Calculate residuals (prediction - actual)
    residuals = [prediction - actual for prediction, actual in zip(predictions, prices)]

    # Calculate gradients
    d_theta0 = sum(residuals) / m
    d_theta1 = (
        sum(residual * mileage for residual, mileage in zip(residuals, mileages)) / m
    )

    # Update parameters
    new_theta0 = theta0 - learning_rate * d_theta0
    new_theta1 = theta1 - learning_rate * d_theta1

    return new_theta0, new_theta1


def train_with_gradient_descent(
    mileages: list[float],
    prices: list[float],
    learning_rate: float = 0.01,
    max_epochs: int = 1000,
    tolerance: float = 1e-6,
) -> tuple[float, float, list[float], list[float], list[float]]:
    """
    Train linear regression model using gradient descent.

    Args:
        mileages: List of mileage values (normalized)
        prices: List of price values (normalized)
        learning_rate: Learning rate for gradient descent
        max_epochs: Maximum number of training epochs
        tolerance: Convergence tolerance for early stopping

    Returns:
        Tuple of (final_theta0, final_theta1, theta0_history, theta1_history, cost_history)
    """
    # Initialize parameters
    theta0 = 0.0
    theta1 = 0.0

    # History tracking
    theta0_history = [theta0]
    theta1_history = [theta1]
    cost_history = []

    previous_cost = float("inf")

    for epoch in range(max_epochs):
        # Perform gradient descent step
        theta0, theta1 = gradient_descent_step(
            theta0, theta1, mileages, prices, learning_rate
        )

        # Track parameter history
        theta0_history.append(theta0)
        theta1_history.append(theta1)

        # Calculate current cost (MSE)
        predictions = [
            linear_prediction(theta1, mileage, theta0) for mileage in mileages
        ]

        # Using the same MSE formula as in utils.py
        squared_errors = [
            (actual - prediction) ** 2
            for actual, prediction in zip(prices, predictions)
        ]
        current_cost = sum(squared_errors) / (2 * len(squared_errors))
        cost_history.append(current_cost)

        # Check for convergence
        if abs(previous_cost - current_cost) < tolerance:
            print(f"Model converged at epoch {epoch + 1}")
            break

        # Check for divergence
        if current_cost > previous_cost:
            print(f"Model may be diverging at epoch {epoch + 1}")
            # Continue training but warn about potential divergence

        previous_cost = current_cost

    return theta0, theta1, theta0_history, theta1_history, cost_history
