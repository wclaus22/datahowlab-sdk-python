"""
Exceptions Module

This module defines custom exception classes for use throughout the project.

Exceptions:
    - InvalidSpectraException: Exception raised when spectra used for prediction are not valid
    - InvalidInputsException: Exception raised when Inputs used for prediction are not valid
    - ModelPredictionException: Exception raised when Model Prediction fails
    - InvalidTimestampsException: Exception raised when timestamps used for prediction are not valid
"""


class InvalidSpectraException(Exception):
    """Exception raised when spectra used for prediction are not valid"""

    def __init__(self, message="The Spectra used for prediction are not valid."):
        self.message = message
        super().__init__(self.message)


class InvalidInputsException(Exception):
    """Exception raised when Inputs used for prediction are not valid"""

    def __init__(self, message="The Inputs used for prediction are not valid."):
        self.message = message
        super().__init__(self.message)


class ModelPredictionException(Exception):
    """Exception raised when Model Prediction fails"""

    def __init__(self, message="Model is not valid for prediction."):
        self.message = message
        super().__init__(self.message)


class InvalidTimestampsException(Exception):
    """Exception raised when timestamps used for prediction are not valid"""

    def __init__(self, message="The timestamps used for prediction are not valid."):
        self.message = message
        super().__init__(self.message)


class PredictionRequestException(Exception):
    """Exception raised when prediction request fails"""

    def __init__(self, message="Prediction request failed."):
        self.message = message
        super().__init__(self.message)
