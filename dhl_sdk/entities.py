# pylint: disable=no-member
# pylint: disable=unsubscriptable-object
"""API Entities Module

This module provides a comprehensive set of Pydantic models that represent
multiple entities obtained from the API . 

Classes:
    - Variable: Represents a structure for variables present in the models.
    - Dataset: Represents a structure for datasets present in the models.
    - Model: Represents a structure for models fetched from the API.
    - Project: Represents a structure for projects retrieved from the API.
"""

from abc import ABC, abstractmethod
from typing import Literal, Optional, Type
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from dhl_sdk.crud import Result, Client, CRUDClient
from dhl_sdk.exceptions import (
    InvalidInputsException,
    ModelPredictionException,
    PredictionRequestException,
)
from dhl_sdk._utils import Predictions, PredictResponse, get_id_list
from dhl_sdk._input_processing import (
    SpectraData,
    GroupCode,
    CultivationPropagationPreprocessor,
    CultivationHistoricalPreprocessor,
    Preprocessor,
    SpectraPreprocessor,
    format_predictions,
)

PROJECTS_URL = "api/db/v2/projects"
DATASETS_URL = "api/db/v2/datasets"
VARIABLES_URL = "api/db/v2/variables"
MODELS_URL = "api/db/v2/pipelineJobs"
TEMPLATES_URL = "api/db/v2/pipelineJobTemplates"
PREDICT_URL = "api/pipeline/v1/predictors"


class Variable(BaseModel):
    """Model Variable"""

    id: str = Field(alias="id")
    name: str = Field(alias="name")
    code: str = Field(alias="code")
    variant: str = Field(alias="variant")
    group_code: Optional[GroupCode] = None
    size: Optional[int] = None

    @model_validator(mode="before")
    @classmethod
    def _validate_model_struct(cls, data) -> dict:
        """Validate the structure of the variable"""
        data["group_code"] = data.get("group", {}).get("code", None)

        if data["variant"] == "spectrum":
            try:
                size = data["spectrum"]["xAxis"]["dimension"]
            except KeyError as err:
                raise KeyError(
                    "The spectrum variable does not have a valid structure"
                ) from err

            data["size"] = size

        return data

    def matches_key(self, key) -> bool:
        """Find the id of the variable"""
        if self.id == key or (self.code is not None and self.code == key):
            return True
        return False

    def __str__(self) -> str:
        """Print only the variable's ID and Code"""
        return f"Name: {self.name} ,  Code: {self.code}"

    @staticmethod
    def requests(client: Client) -> CRUDClient["Variable"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["Variable"](client, VARIABLES_URL, Variable)


class Dataset(BaseModel):
    """Model Dataset"""

    id: str = Field(alias="id")
    variables: list[Variable] = Field(alias="variables")
    _client: Client = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    @model_validator(mode="before")
    @classmethod
    def _validate_variables(cls, data) -> dict:
        unpacked_variables = []

        for i, variable_info in enumerate(data["variables"]):
            try:
                var_id = variable_info["id"]
            except KeyError as err:
                raise KeyError(
                    f"The variable at index {i} does not contain an id"
                ) from err

            var = Variable.requests(data["client"]).get(var_id)
            unpacked_variables.append(var)

        data["variables"] = unpacked_variables

        return data

    @staticmethod
    def requests(client: Client) -> CRUDClient["Dataset"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["Dataset"](client, DATASETS_URL, Dataset)


class SpectraDataset(Dataset):
    """Pydantic Model Dataset for Spectra"""

    _client: Client = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    def get_spectrum_index(self) -> int:
        """Get the index of the spectrum variable"""
        for index, variable in enumerate(self.variables):
            if variable.variant == "spectrum":
                return index
        raise ValueError("No spectrum variable found in dataset")

    @staticmethod
    def requests(client: Client) -> CRUDClient["SpectraDataset"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["SpectraDataset"](client, DATASETS_URL, SpectraDataset)


class Model(BaseModel, ABC):
    """Pydantic BaseModel for predictive models from the API"""

    id: str = Field(alias="id")
    name: str = Field(alias="name")
    status: str = Field(alias="status")
    project_id: str = Field(alias="projectId")
    config: dict = Field(alias="config")
    _client: Client = PrivateAttr()

    @property
    def success(self) -> bool:
        """Get the success status of the model"""
        return self.status == "success"

    @staticmethod
    @abstractmethod
    def requests(client: Client) -> CRUDClient["Model"]:
        """Resquests abstract method for Model Types"""

    def get_predictions(self, preprocessor: Preprocessor) -> dict:
        """Get the predictions for the model using selected strategy"""

        if preprocessor.validate():
            json_data = preprocessor.format()
        else:
            raise InvalidInputsException(
                "The provided inputs failed the validation step"
            )

        predict_url = f"{PREDICT_URL}/{self.id}/predict"

        predictions = []
        for prediction_data in json_data:
            try:
                response = self._client.post(predict_url, prediction_data)
                response.raise_for_status()

                # in case of an error in the response (not HTTP)
                if "error" in response.json():
                    raise PredictionRequestException(response.json()["error"])

            except Exception as ex:
                raise ex

            predictions.append(PredictResponse(**response.json()))

        return format_predictions(predictions, model=self)


class SpectraModel(Model):
    """Pydantic Model for Spectra Prediction Model from the API"""

    dataset: SpectraDataset = Field(alias="dataset")
    _spectra_size: Optional[int] = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    @model_validator(mode="before")
    @classmethod
    def _validate_model_data(cls, data) -> dict:
        data["dataset"] = SpectraDataset(**data["dataset"], client=data["client"])
        return data

    def predict(
        self,
        spectra: SpectraData,
        inputs: Optional[dict] = None,
    ) -> Predictions:
        """
        Predicts the output of a given model for a given set of spectra.

        Parameters:
        -----------
        spectra : list[list[float]] or np.ndarray
            A 2D array representing spectra for prediction, where:
                - The first dimension corresponds to the spectra index.
                - The second dimension contains wavelength index.
        inputs : dict, optional
            Additional inputs to be used for prediction. The keys must be the Codes of the
            input variables, and the values must be lists of the same length as the number
            of spectrum, by default None.

        Returns:
        --------
        Dictionary with predictions where:
            key: variable code
            value: list with predictions for each spectrum

        Example:
        --------
        >>> spectra = [[1.0, 2.0, 3.0, 3.0, 4.0, 3.0], [2.0, 3.0, 4.0, 4.0, 5.0, 4.0]]
        >>> inputs = {"input1": [42, 33]}
        >>> result = model.predict(spectra, inputs)
        >>> print(result)
        {'output1': [prediction1, prediction2], 'output2': [prediction1, prediction2]}

        """

        if not self.success:
            raise ModelPredictionException(
                f"{self.name} is not ready for prediction. The current status is {self.status}"
            )

        spectra_processing_strategy = SpectraPreprocessor(
            spectra=spectra, inputs=inputs, model=self
        )

        return super().get_predictions(spectra_processing_strategy)

    @property
    def inputs(self) -> list[str]:
        """Get the inputs from the model's config"""
        return self.config["groups"]["Inputs"]

    @property
    def outputs(self) -> list[str]:
        """Get the outputs from the model's config"""
        return self.config["groups"]["Outputs"]

    @property
    def spectra_size(self) -> int:
        """Get the size of the spectra"""
        if self._spectra_size is None:
            self._spectra_size = self._get_spectra_size()
        return self._spectra_size

    def _get_spectra_size(self) -> int:
        """Get the size of the spectra from variable information in the API"""
        spectrum = self.dataset.variables[self.dataset.get_spectrum_index()]
        return spectrum.size

    @staticmethod
    def requests(client: Client) -> CRUDClient["SpectraModel"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["SpectraModel"](client, MODELS_URL, SpectraModel)


class CultivationModel(Model, ABC):
    """Abstract Pydantic Model for Cultivation Prediction Model from the API"""

    dataset: Dataset = Field(alias="dataset")

    @model_validator(mode="before")
    @classmethod
    def _validate_model_data(cls, data) -> dict:
        data["dataset"] = Dataset(**data["dataset"], client=data["client"])
        return data

    @abstractmethod
    def predict(
        self, timestamps: list, inputs: dict, timestamps_unit: str = "s"
    ) -> dict:
        """Prediction for CultivationModel"""

    @staticmethod
    @abstractmethod
    def requests(client: Client) -> CRUDClient["CultivationModel"]:
        """CRUDClient for CultivationModel"""


class CultivationPropagationModel(CultivationModel):
    """Pydantic Model for Propagation Model for Cultivation from the API"""

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    def predict(
        self, timestamps: list, inputs: dict, timestamps_unit: str = "s"
    ) -> dict:
        """
        Predicts the output of a given model for a given set of inputs.

        Parameters:
        -----------
        timestamps : list
            A list of timestamps for prediction.
        inputs : dict, optional
            Inputs to be used for prediction. The keys must be the Codes of the
            input variables, and the values must be lists of the same length as the timestamps.
        timestamps_unit : str, optional
            Unit of the timestamps, by default "s".
            Needs to be one of the following: "s", "m", "h", "d".

        Returns:
        --------
        Dictionary with predictions where:
            key: variable code
            value: list with predictions for each spectrum

        Example:
        --------
        >>> timestamps = [1,2,3,4,5,6,7]
        >>> inputs = {"var1": [42], "var2": [0.3], "var3": [0.5], "var4": [0,2,3,3,3,3,3]}
        >>> result = model.predict(timestamps, inputs, timestamps_unit="d")
        >>> print(result)
        {'var2': [pred1, pred2, pred3, pred4, pred5, pred6, pred7],
        'var3': [pred1, pred2, pre3, pred4, pred5, pred6, pred7]}

        """

        if not self.success:
            raise ModelPredictionException(
                f"{self.name} is not ready for prediction. The current status is {self.status}"
            )

        data_processing_strategy = CultivationPropagationPreprocessor(
            timestamps=timestamps,
            timestamps_unit=timestamps_unit,
            inputs=inputs,
            model=self,
        )

        return super().get_predictions(data_processing_strategy)

    @staticmethod
    def requests(client: Client) -> CRUDClient["CultivationPropagationModel"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["CultivationPropagationModel"](
            client, MODELS_URL, CultivationPropagationModel
        )


class CultivatioHistoricalModel(CultivationModel):
    """Pydantic Model for Historical Model for Cultivation from the API"""

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    def predict(
        self, timestamps: list, inputs: dict, timestamps_unit: str = "s"
    ) -> dict:
        """
        Predicts the output of a given model for a given set of inputs.

        Parameters:
        -----------
        timestamps : list
            A list of timestamps for prediction.
        inputs : dict, optional
            Inputs to be used for prediction. The keys must be the Codes of the
            input variables, and the values must be lists of the same length as the timestamps.
        timestamps_unit : str, optional
            Unit of the timestamps, by default "s".
            Needs to be one of the following: "s", "m", "h", "d".

        Returns:
        --------
        Dictionary with predictions where:
            key: variable code
            value: list with predictions for each spectrum

        Example:
        --------
        >>> timestamps = [1,2,3,4,5,6,7]
        >>> inputs = {"var1": [42], "var2": [0.3], "var3": [0.5], "var4": [0,2,3,3,3,3,3]}
        >>> result = model.predict(timestamps, inputs, timestamps_unit="d")
        >>> print(result)
        {'output1': [pred1], 'output2': [pred2]}

        """

        if not self.success:
            raise ModelPredictionException(
                f"{self.name} is not ready for prediction. The current status is {self.status}"
            )

        data_processing_strategy = CultivationHistoricalPreprocessor(
            timestamps=timestamps,
            timestamps_unit=timestamps_unit,
            inputs=inputs,
            model=self,
        )

        return super().get_predictions(data_processing_strategy)

    @staticmethod
    def requests(client: Client) -> CRUDClient["CultivatioHistoricalModel"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["CultivatioHistoricalModel"](
            client, MODELS_URL, CultivatioHistoricalModel
        )


class ModelFactory:
    """Factory for Model, given the process unit id and model type"""

    MODEL_MAP = {
        "373c173a-1f23-4e56-874e-90ca4702ec0d": SpectraModel,
        "04a324da-13a5-470b-94a1-bda6ac87bb86": CultivationModel,
    }

    def __init__(self, process_unit_id):
        self._process_unit_id = process_unit_id

    def get_model(self, **kwargs) -> Type[Model]:
        """Get the model type from the process unit id"""

        if self._process_unit_id not in self.MODEL_MAP:
            raise NotImplementedError(
                f"Process unit id {self._process_unit_id} is not currently supported"
            )

        model = self.MODEL_MAP[self._process_unit_id]

        if "model_type" in kwargs and model == CultivationModel:
            if kwargs["model_type"] == "propagation":
                model = CultivationPropagationModel
            elif kwargs["model_type"] == "historical":
                model = CultivatioHistoricalModel

        return model


class Project(BaseModel, ABC):
    """Abstract class for a DHL Project"""

    id: str = Field(alias="id")
    name: str = Field(alias="name")
    description: str = Field(alias="description")
    process_unit_id: str = Field(alias="processUnitId")
    _client: Client = PrivateAttr()

    @abstractmethod
    def get_models(self, model_name: Optional[str] = None) -> Result[Model]:
        """Get the models of the project from the API"""

    @abstractmethod
    def _get_model_query_params(
        self, model_name: Optional[str] = None
    ) -> dict[str, str]:
        """Get the query params for the models"""

    @staticmethod
    @abstractmethod
    def requests(client: Client) -> CRUDClient["Project"]:
        """Resquests abstract method for Project Types"""


class SpectraProject(Project):
    """Pydantic Model for a DHL Project from the API"""

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    def get_models(self, model_name: Optional[str] = None) -> Result[Model]:
        """Get the models of the project from the API"""

        model = ModelFactory(self.process_unit_id).get_model()

        models = model.requests(self._client)
        query_params = self._get_model_query_params(model_name=model_name)

        results = Result[model](
            limit=5,
            query_params=query_params,
            requests=models,
        )

        return results

    def _get_model_query_params(
        self, model_name: Optional[str] = None
    ) -> dict[str, str]:
        query_params = {"filterBy[projectId]": self.id}

        if model_name is not None:
            query_params.update({"filterBy[name]": model_name})

        return query_params

    @staticmethod
    def requests(client: Client) -> CRUDClient["SpectraProject"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["SpectraProject"](client, PROJECTS_URL, SpectraProject)


class CultivationProject(Project):
    """Pydantic Model for a DHL Project from the API"""

    id: str = Field(alias="id")
    name: str = Field(alias="name")
    description: str = Field(alias="description")
    process_unit_id: str = Field(alias="processUnitId")
    _client: Client = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = data["client"]

    def get_models(
        self,
        model_name: Optional[str] = None,
        model_type: Literal["propagation", "historical"] = "propagation",
    ) -> Result[Model]:
        """Get the models of the project from the API"""

        if model_type not in ["propagation", "historical"]:
            raise ValueError(
                f"model_type must be either propagation or historical, got {model_type}"
            )

        query_params = self._get_model_query_params(
            model_name=model_name, model_type=model_type
        )

        model = ModelFactory(self.process_unit_id).get_model(model_type=model_type)
        models = model.requests(self._client)

        results = Result[model](
            limit=5,
            query_params=query_params,
            requests=models,
        )
        return results

    def _get_model_query_params(
        self,
        model_name: Optional[str] = None,
        model_type: Literal["propagation", "historical"] = "propagation",
    ) -> dict[str, str]:
        query_params = {"filterBy[projectId]": self.id}

        if model_name is not None:
            query_params.update({"filterBy[name]": model_name})

        # get templateIds for propagation models
        template_query_params = {
            "filterByTag[type]": model_type,
            "archived": "any",
        }
        template_list = self._client.get(TEMPLATES_URL, template_query_params).json()
        template_ids = get_id_list(template_list)

        query_params.update({"filterBy[templateId]": "|".join(template_ids)})

        return query_params

    @staticmethod
    def requests(client: Client) -> CRUDClient["CultivationProject"]:
        # pylint: disable=missing-function-docstring
        return CRUDClient["CultivationProject"](
            client, PROJECTS_URL, CultivationProject
        )
