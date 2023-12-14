"""This module contains utility functions for data validation and formatting in the SDK
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Protocol, Union

import numpy as np

from dhl_sdk._utils import (
    PredictResponse,
    validate_list_elements,
)
from dhl_sdk.exceptions import (
    InvalidSpectraException,
    InvalidInputsException,
    InvalidTimestampsException,
)
from dhl_sdk._spectra_utils import (
    SpectraData,
    SpectraPrediction,
    SpectraModel,
    _validate_spectra_format,
    _convert_to_request,
)


class GroupCode(str, Enum):
    """Enum for the group code"""

    # Groups for Upstream Models
    FLOWS = "Flows"
    FEED_CONC = "FeedConc"
    INDUCER = "Inducer"
    W = "W"
    Z = "Z"
    X = "X"
    Y = "Y"

    # Groups for Spectra Models
    SPC = "SPC"
    TGT = "TGT"

    def is_timedependent(self):
        """Check if the group is time dependent
        for recipe only"""
        return self in [GroupCode.FLOWS, GroupCode.W, GroupCode.INDUCER]

    def is_output(self):
        """Check if the group is an output / CQA"""
        return self == GroupCode.Y


class CultivationModel(Protocol):
    # pylint: disable=missing-class-docstring
    # pylint: disable=missing-function-docstring
    @property
    def dataset(self) -> list[str]:
        ...


# Input Preprocessors (Validation and Formatting)
class PreprocessorStrategy(ABC):
    """Interface for the preprocessing strategy"""

    @abstractmethod
    def validate(self) -> bool:
        """Validate the input data"""

    @abstractmethod
    def format(self) -> list[dict]:
        """Format the input data to a list of JSON requests"""


@dataclass
class SpectraPreprocessor(PreprocessorStrategy):
    """Implementation of the preprocessing strategy for Spectrum models"""

    spectra: SpectraData
    inputs: Optional[dict]
    model: SpectraModel

    def validate(self) -> bool:
        """
        This function validates and makes the necessary formating of the spectra
        and inputs used for prediction. It performs the following validations:

        - Validates if spectra is empty
        - Validates if the number of wavelengths in spectra matches
            the number of wavelengths in the model
        - Validates if the spectra contains None values
        - Validates if the number of inputs matches model inputs
        - Validates if the number of values in each input matches the number of spectra

        Parameters
        ----------
        spectra : Union[list[list[float]], np.ndarray]
            The spectra to be validated
        model : Model
            The model to use for prediction.
        inputs : dict, optional
            A dictionary of input variables and their values, by default None

        Returns
        -------
        list[list[float], dict
            If all validations pass, return the formatted spectra and inputs
            as a tuple.

        Raises
        ------
        InvalidSpectraException
            Exception raised when Spectra is not valid for prediction.
        InvalidInputsException
            Exception raised when Inputs is not valid for prediction.
        """

        # Validate if empty spectra
        n_spectra = len(self.spectra)
        if n_spectra < 1:
            raise InvalidSpectraException("Empty spectra provided")

        self.spectra = _validate_spectra_format(self.spectra)

        # Validate number of wavelengths in spectra
        for i, spectrum in enumerate(self.spectra):
            if len(spectrum) != self.model.spectra_size:
                raise InvalidSpectraException(
                    (
                        f"Invalid Spectra: The Number of Wavelengths does not"
                        f"match training data for spectrum number: {i+1}. "
                        f"Expected: {self.model.spectra_size}, Got: {len(spectrum)}"
                    )
                )
            if validate_list_elements(spectrum):
                raise InvalidSpectraException(
                    (
                        f"Invalid Spectra: The Spectra contains not "
                        f"valid values for spectrum number: {i+1}"
                    )
                )

        if self.inputs is None:
            if len(self.model.inputs) > 0:
                raise InvalidInputsException(
                    "The model requires inputs, but none were provided."
                )
            return True

        if len(self.model.inputs) == 0:
            raise InvalidInputsException(
                "The model does not require inputs, but some were provided."
            )

        # validate inputs with spectra inputs (number of lines)
        for key, value in self.inputs.items():
            if len(value) != n_spectra:
                raise InvalidInputsException(
                    f"The Number of values does not match the number of spectra for input: {key}"
                )
            if validate_list_elements(value):
                raise InvalidInputsException(
                    f"Invalid Inputs: The Inputs contains not valid values for input: {key}"
                )

        return True

    def format(self) -> list[dict]:
        """
        Format the inputs for a given model.

        Returns
        -------
        list[dict]
            A list of JSON requests, where each request contains a list of instances.

        Raises
        ------
        InvalidInputsException
            If no matching input is found for a given key.
        """

        if self.inputs is None:
            return _convert_to_request(self.spectra, model=self.model)

        model_variables = self.model.dataset.variables

        input_variables = [
            variable for variable in model_variables if variable.id in self.model.inputs
        ]

        # validate inputs codes and format for ids
        formatted_inputs = {}
        for key, value in self.inputs.items():
            for variable in input_variables:
                if variable.matches_key(key):
                    formatted_inputs[variable.id] = value
                    break
            else:
                correct_inputs = [print(variable) for variable in input_variables]
                raise InvalidInputsException(
                    (
                        f"No matching Input found for key: {key}. "
                        f"Please select one of the following as inputs: {*correct_inputs,}"
                    )
                )

        return _convert_to_request(
            self.spectra, model=self.model, inputs=formatted_inputs
        )


@dataclass
class CultivationPreprocessor(PreprocessorStrategy):
    """Implementation of the preprocessing strategy for cultivation models"""

    timestamps: list[int]
    timestamps_unit: str
    inputs: Optional[Union[list[dict], dict]]
    model: CultivationModel

    def validate(self) -> bool:
        """
        This method validates the timestamps and inputs used for prediction.
        It performs the following validations:

        - Validates the structure and values of the timestamps
            - Valid timestamps should be a list of integers, have a length of at least 2,
            contain valid numeric values,be in ascending order, have positive values and be unique.
        - Validates the structure and values of the inputs
            - The inputs should be organized with variable codes as \
                keys and lists of input values as values.
        - Validates the inputs based on the model variables
            - Garantees that all the mandatory model variables are present in the inputs
            - Garantees that the inputs that need to be complete for the recipe \
                have the same length as the timestamps
            - Garantees that the initial condition variables only have one value 

        Returns
        -------
        bool
            True if all validations pass

        Raises
        ------
        InvalidTimestampsException
            Exception raised when Spectra is not valid for prediction.
        InvalidInputsException
            Exception raised when Inputs is not valid for prediction.
        """

        if isinstance(self.timestamps, np.ndarray):
            self.timestamps = self.timestamps.tolist()

        # Validate timestamps
        self.timestamps = _validate_upstream_timestamps(
            timestamps=self.timestamps, timestamps_unit=self.timestamps_unit
        )

        # Validate inputs
        _validate_upstream_inputs(inputs=self.inputs)

        # validate inputs with model variables
        _validate_with_variables(self.timestamps, self.inputs, self.model)
        return True

    def format(self) -> list[dict]:
        formatted_inputs = {}
        input_variables = self.model.dataset.variables

        # order the dict according to Variables and insert timestamps
        for key, value in self.inputs.copy().items():
            for variable in input_variables:
                if variable.matches_key(key):
                    formatted_inputs[variable.id] = {}
                    formatted_inputs[variable.id]["values"] = value
                    formatted_inputs[variable.id]["timestamps"] = self.timestamps[
                        : len(value)
                    ]
                    break

        instances = []

        if not isinstance(self.inputs, list):
            self.inputs = [self.inputs]

        for i, _ in enumerate(self.inputs):
            instances.append([])
            for variable in input_variables:
                if variable.id in formatted_inputs:
                    instances[i].append(formatted_inputs[variable.id])
                else:
                    instances[i].append({"values": None, "timestamps": None})

        json_data = {"instances": instances, "config": {"startingIndex": 0}}

        return [json_data]


class Preprocessor:
    """Context class for the Preprocessor strategy"""

    def __init__(self, strategy: PreprocessorStrategy) -> None:
        self._strategy = strategy

    def validate(self) -> bool:
        """Validates the data"""
        return self._strategy.validate()

    def convert_to_request(self) -> list[dict]:
        """Converts the data to a list of JSON requests"""
        if self._strategy.validate():
            return self._strategy.format()


def _validate_upstream_timestamps(
    timestamps: list[Union[int, float]], timestamps_unit: str
) -> list[int]:
    """Validate the timestamps for upstream prediction.

    This function performs a series of validations on the provided timestamps to ensure they meet
    the requirements for upstream prediction tasks.

    Parameters
    ----------
    timestamps : list[int]
        List of timestamps

    """
    # Validate types
    if not isinstance(timestamps, list):
        raise InvalidTimestampsException("Timestamps must be a list of numbers")

    # Validate if length of timestamps
    if len(timestamps) <= 1:
        raise InvalidTimestampsException(
            "Timestamps must be a list of at least 2 values"
        )

    # Validate if timestamps are valid numeric values
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in timestamps
    ):
        raise InvalidTimestampsException(
            "All values of timestamps must be valid numeric values"
        )

    # Validate if timestamps are in ascending order
    if all(timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)):
        raise InvalidTimestampsException("Timestamps must be in ascending order")

    # Validate if timestamps are positive (Since they are ordered, just check the first one)
    if timestamps[0] < 0:
        raise InvalidTimestampsException("Timestamps must be positive")

    # Validate if timestamps are unique
    if len(timestamps) != len(set(timestamps)):
        raise InvalidTimestampsException("Timestamps must be unique")

    # Convert timestamps to seconds
    if timestamps_unit in ("s", "sec", "secs", "seconds"):
        return timestamps
    elif timestamps_unit in ("m", "min", "mins", "minutes"):
        factor = 60
        return [timestamp * factor for timestamp in timestamps]
    elif timestamps_unit in ("h", "hour", "hours"):
        factor = 60 * 60
        return [timestamp * factor for timestamp in timestamps]
    elif timestamps_unit in ("d", "day", "days"):
        factor = 60 * 60 * 24
        return [timestamp * factor for timestamp in timestamps]
    else:
        raise InvalidTimestampsException(
            f"Invalid timestamps unit '{timestamps_unit}' found."
        )


def _validate_upstream_inputs(inputs: Union[dict[str, list], list[dict[str, list]]]):
    """
    Validate the inputs for upstream prediction.
    The function checks if the inputs are a dictionary and if all values are lists.

    Parameters
    ----------
    inputs : dict[str, list]
        A dictionary, or list of dictionaries, where keys are variable codes,
        and values are lists of inputs.

    Raises
    ------
    InvalidInputsException
        If inputs do not meet the specified criteria.
    """

    if inputs is None:
        raise InvalidInputsException(
            "No Inputs provided. Please provide a dictionary of inputs"
        )

    def validate_single_input(single_input: dict[str, list]):
        # Validate types
        if not isinstance(single_input, dict):
            raise InvalidInputsException(
                "Inputs must be a dictionary of lists, with the variable code as key"
            )

        # Validate if inputs are lists
        if not all(isinstance(value, list) for value in single_input.values()):
            raise InvalidInputsException("All input values must be lists")

    if isinstance(inputs, list):
        for single_input in inputs:
            validate_single_input(single_input)
    else:
        validate_single_input(inputs)


def _validate_with_variables(
    timestamps: list[int],
    inputs: Union[dict[str, list], list[dict[str, list]]],
    model: CultivationModel,
):
    """
    Validate the inputs based on the provided timestamps and model.

    Parameters
    ----------
    timestamps : list[int]
        List of timestamps
    inputs : dict[str, list]
        A dictionary where keys are variable codes, and values are lists of inputs.
    model : Model
        Model used for prediction

    Raises
    ------
    InvalidInputsException
        If inputs do not meet the specified criteria.
    """

    model_variables = model.dataset.variables

    def validate_input_vars(
        single_input: dict[str, list], model_variables=model_variables
    ):
        for variable in model_variables:
            if variable.code in inputs:
                if variable.group_code.is_timedependent():
                    # Validate if inputs and timestamps are the same length
                    if len(single_input[variable.code]) != len(timestamps):
                        raise InvalidInputsException(
                            (
                                f"The recipe requires {variable.code} to be complete,"
                                f"so it must have the same length as timestamps"
                            )
                        )

                    # Validate if inputs are valid numeric values
                    if not all(
                        isinstance(value, (int, float)) and math.isfinite(value)
                        for value in single_input[variable.code]
                    ):
                        raise InvalidInputsException(
                            f"All values of input {variable.code} must be valid numeric values"
                        )

                else:
                    # validate if non time dependent inputs have length of 1 (only initial values)
                    if len(single_input[variable.code]) != 1:
                        raise InvalidInputsException(
                            (
                                f"Input {variable.code} is only requires initial "
                                f"values, so it must have a length of 1"
                            )
                        )

            else:
                # validate if missing value is an Y Variable
                if not variable.group_code.is_output():
                    raise InvalidInputsException(
                        (
                            f"Input {variable.code} is a {variable.group_code.name} "
                            f"Variable, so it must be provided"
                        )
                    )

    if isinstance(inputs, list):
        for single_input in inputs:
            validate_input_vars(single_input)
    else:
        validate_input_vars(inputs)


def format_predictions(
    predictions: list[PredictResponse], model: SpectraModel
) -> SpectraPrediction:
    """Format a list of predictions into a dictionary.

    Parameters
    ----------
    predictions : List[PredictResponse]
        list of predictions from the API.
    model : Model
        Model used for prediction

    Returns
    -------
    Dictionary with predictions where:
        key: variable id
        value: list of predictions
    """
    variables = [var.code for var in model.dataset.variables]

    dic = {}

    for pred in predictions:
        for i, instance in enumerate(pred.instances[0]):
            if instance is not None:
                if variables[i] in dic:
                    dic[variables[i]].extend(instance.values)
                else:
                    dic[variables[i]] = instance.values.copy()

    return dic
