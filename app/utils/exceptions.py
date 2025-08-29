"""
Custom exceptions for Vanna.AI implementation.
"""


class VannaException(Exception):
    """Base exception class for Vanna.AI."""
    pass


class DatabaseConnectionError(VannaException):
    """Raised when database connection fails."""
    pass


class LLMError(VannaException):
    """Raised when LLM API calls fail."""
    pass


class TrainingDataError(VannaException):
    """Raised when training data is invalid or corrupted."""
    pass


class SQLGenerationError(VannaException):
    """Raised when SQL generation fails."""
    pass


class VectorStoreError(VannaException):
    """Raised when vector store operations fail."""
    pass


class ConfigurationError(VannaException):
    """Raised when configuration is invalid."""
    pass 