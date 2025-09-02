"""Custom exceptions for data handling."""


class InvalidCsvException(Exception):
    """Custom exception for handling invalid CSV fields."""

    def __init__(self, message: str) -> None:
        """
        Initialize InvalidCsvException.

        Args:
            message: Error message describing the CSV issue
        """
        self.message = message
        super().__init__(message)


class InvalidArgException(Exception):
    """Custom exception for handling invalid function arguments."""

    def __init__(self, message: str) -> None:
        """
        Initialize InvalidArgException.

        Args:
            message: Error message describing the argument issue
        """
        self.message = message
        super().__init__(message)
