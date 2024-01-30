"""This module contains generic utility functions used in the SDK
"""

from functools import reduce
import urllib.parse as urlparse
from typing import Optional, Union
import numpy as np
from pydantic import BaseModel, Field, model_validator


Predictions = dict[str, list[float]]


class Instance(BaseModel):
    """Pydantic class representing the Instance
    It is used to type check the request"""

    timestamps: Optional[list[float]] = Field(default=None, alias="timestamps")
    sample_id: Optional[list[str]] = Field(default=None, alias="sampleId")
    values: Union[list[float], list[list[float]]]
    high_values: Optional[Union[list[float], list[list[float]]]] = Field(
        default=None, alias="highValues"
    )
    low_values: Optional[Union[list[float], list[list[float]]]] = Field(
        default=None, alias="lowValues"
    )

    @model_validator(mode="after")
    def generate_sample_ids(self):
        """Generates sample ids if not provided"""
        if self.sample_id is None:
            self.sample_id = [str(i) for i in range(len(self.values))]
        return self


class PredictRequest(BaseModel):
    """Pydantic class representing the expected Predict Request"""

    instances: list[list[Optional[Instance]]]
    metadata: Optional[dict] = None
    config: Optional[dict] = None


class PredictResponse(BaseModel):
    """Pydantic class representing the expected Predict Response"""

    instances: list[list[Optional[Instance]]]


def urljoin(*args) -> str:
    """join url elements together into one url"""
    elements = [
        f"{arg}/" if arg[-1] != "/" and i != len(args) - 1 else arg
        for i, arg in enumerate(args)
    ]
    return reduce(urlparse.urljoin, elements)


def validate_list_elements(arr: list) -> bool:
    """Validates if an array contains non float values"""
    return any(
        not isinstance(item, (float, int))
        or item is None
        or np.isnan(item)
        or np.isinf(item)
        for item in arr
    )


def get_id_list(json_list: list[dict]) -> list[str]:
    """Extracts the id from a list of dictionaries"""
    return [item["id"] for item in json_list]
