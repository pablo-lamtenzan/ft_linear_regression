"""Mathematical utilities for linear regression."""


def normalize(x: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to the range [0, 1].

    Args:
        x: Value to normalize
        min_val: Minimum value in the dataset
        max_val: Maximum value in the dataset

    Returns:
        Normalized value between 0 and 1

    Raises:
        ValueError: If max_val equals min_val (cannot normalize)
    """
    if max_val == min_val:
        raise ValueError("Maximum and minimum values are equal; cannot normalize.")
    return (x - min_val) / (max_val - min_val)


def denormalize(x: float, min_val: float, max_val: float) -> float:
    """
    Denormalize a value from the range [0, 1] back to original scale.

    Args:
        x: Normalized value between 0 and 1
        min_val: Minimum value in the original dataset
        max_val: Maximum value in the original dataset

    Returns:
        Denormalized value in original scale
    """
    return x * (max_val - min_val) + min_val


def calculate_mse(actuals: list[float], predictions: list[float]) -> float:
    """
    Calculate Mean Squared Error between actual and predicted values.

    Args:
        actuals: List of actual values
        predictions: List of predicted values

    Returns:
        Mean squared error value

    Raises:
        ValueError: If lists have different lengths or are empty
    """
    if len(actuals) != len(predictions):
        raise ValueError(
            "calculate_mse: The length of actuals and predictions must be the same."
        )

    if len(actuals) == 0:
        raise ValueError("calculate_mse: Cannot calculate MSE for empty lists.")

    squared_errors = [
        (actual - prediction) ** 2 for actual, prediction in zip(actuals, predictions)
    ]
    return sum(squared_errors) / (2 * len(squared_errors))


def linear_prediction(slope: float, x: float, y_intercept: float) -> float:
    """
    Calculate linear prediction using the equation: y = slope * x + y_intercept.

    Args:
        slope: Slope parameter (theta1)
        x: Input value (mileage)
        y_intercept: Y-intercept parameter (theta0)

    Returns:
        Predicted value
    """
    return slope * x + y_intercept
