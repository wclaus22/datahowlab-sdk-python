import unittest
from unittest.mock import Mock

import numpy as np
from pydantic import BaseModel
from dhl_sdk.entities import Variable
from dhl_sdk.crud import Result

from dhl_sdk._utils import (
    Instance,
    PredictResponse,
)
from dhl_sdk.exceptions import InvalidSpectraException, InvalidInputsException
from dhl_sdk._input_processing import (
    format_predictions,
    _validate_spectra_format,
    Preprocessor,
    SpectraPreprocessor,
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        self.model_no_inputs = Mock()
        self.model_no_inputs.inputs = []
        self.model_with_inputs = Mock()
        self.model_with_inputs.dataset.variables = [
            Variable(id="id-123", code="var1", variant="numeric", name="variable 1"),
            Variable(id="id-456", code="var2", variant="numeric", name="variable 2"),
            Variable(id="id-789", code="out1", variant="numeric", name="output 1"),
            Variable(id="id-101", code="out2", variant="numeric", name="output 2"),
        ]
        self.model_with_inputs.inputs = ["id-123", "id-456"]

    # def test_format_inputs(self):
    # inputs = {"var1": [1, 2, 3], "var2": [4, 5, 6]}
    # formatted_inputs = format_inputs(
    #     inputs, self.model_with_inputs, self.model_with_inputs.inputs
    # )

    # self.assertDictEqual(
    #     formatted_inputs, {"id-123": [1, 2, 3], "id-456": [4, 5, 6]}
    # )

    def test_format_spectra_validation(self):
        spectra1 = [[1, 2, 3], [4, 5, 6]]
        spectra2 = np.array([[1, 2, 3], [4, 5, 6]])
        spectra3 = "spectra"

        self.assertEqual(_validate_spectra_format(spectra1), spectra1)
        self.assertEqual(_validate_spectra_format(spectra2), spectra1)
        self.assertRaises(InvalidSpectraException, _validate_spectra_format, spectra3)

    def test_format_predictions(self):
        predictions = [
            PredictResponse(
                instances=[
                    [
                        None,
                        None,
                        Instance(values=[1, 2, 3]),
                        Instance(values=[4, 5, 6]),
                    ]
                ]
            ),
            PredictResponse(
                instances=[
                    [
                        None,
                        None,
                        Instance(values=[1, 2, 3]),
                        Instance(values=[4, 5, 6]),
                    ]
                ]
            ),
        ]

        formatted_predictions = format_predictions(
            predictions, model=self.model_with_inputs
        )

        self.assertDictEqual(
            formatted_predictions,
            {"out1": [1, 2, 3, 1, 2, 3], "out2": [4, 5, 6, 4, 5, 6]},
        )

    def test_validation_no_input(self):
        model = self.model_no_inputs
        model.spectra_size = 4

        empty_spectra = []
        processor = Preprocessor(
            SpectraPreprocessor(spectra=empty_spectra, model=model, inputs=None)
        )
        self.assertRaises(InvalidSpectraException, processor.validate)

        spectra = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        processor = Preprocessor(
            SpectraPreprocessor(spectra=spectra, model=model, inputs=None)
        )
        self.assertRaises(
            InvalidSpectraException,
            processor.validate,
        )

        spectra = [["1", "2", "3", "3"], [4, 5, 6, 6], [7, 8, 9, 9]]
        processor = Preprocessor(
            SpectraPreprocessor(spectra=spectra, model=model, inputs=None)
        )
        self.assertRaises(
            InvalidSpectraException,
            processor.validate,
        )

        spectra = [[1, 2, 3, 3], [4, 5, np.nan, 6], [7, 8, 9, 9]]
        processor = Preprocessor(
            SpectraPreprocessor(spectra=spectra, model=model, inputs=None)
        )
        self.assertRaises(
            InvalidSpectraException,
            processor.validate,
        )

        spectra = [[1, 2, 3, 3], [4, 5, 6, 6], [7, 8, 9]]
        processor = Preprocessor(
            SpectraPreprocessor(spectra=spectra, model=model, inputs=None)
        )
        self.assertRaises(
            InvalidSpectraException,
            processor.validate,
        )

    # def test_validation_with_input(self):
    #     model = self.model_with_inputs
    #     model.spectra_size = 4

    #     spectra = [[1, 2, 3, 3], [4, 5, 6, 6], [7, 8, 9, 9]]
    #     processor = Preprocessor(
    #         SpectraPreprocessor(spectra=spectra, model=model, inputs=None)
    #     )
    #     self.assertRaises(
    #         InvalidInputsException,
    #         processor.validate,
    #     )

    #     inputs = {"var10": [0, 1, 0], "var2": [1, 1, 1]}
    #     processor = Preprocessor(
    #         SpectraPreprocessor(spectra=spectra, model=model, inputs=inputs)
    #     )
    #     with self.assertRaises(InvalidInputsException) as ex:
    #         processor.validate()
    #         print(ex)
    #         self.assertTrue(
    #             ex.exception.message.startswith(
    #                 "No matching Input found for key: var10"
    #             )
    #         )

    #     inputs = {"var1": [0, 1, 0], "var2": [1, 1, 1]}
    #     processor = Preprocessor(
    #         SpectraPreprocessor(spectra=spectra, model=model, inputs=inputs)
    #     )
    #     self.assertEqual(
    #         processor.validate(),
    #         (spectra, {"id-123": [0, 1, 0], "id-456": [1, 1, 1]}),
    #     )


class TestResults(unittest.TestCase):
    def setUp(self):
        self.client = Mock()

    def test_results(self):
        class DummyEntity(BaseModel):
            id: int
            name: str

        class DummyRequests:
            def list(self, offset, limit, query_params):
                # creates a dummy fetch function that returns 15 elements
                return [
                    DummyEntity(id=i, name=f"Name {i}")
                    for i in range(offset, offset + limit)
                    if i < 15
                ], 15

        results = Result[DummyEntity](
            offset=0,
            limit=5,
            query_params={},
            requests=DummyRequests(),
        )

        self.assertEqual(len(results), 15)

        # tests the first 5 items
        self.assertEqual(next(results).id, 0)
        self.assertEqual(next(results).id, 1)
        self.assertEqual(next(results).id, 2)
        self.assertEqual(next(results).name, "Name 3")
        next(results)

        # tests that the fetch is called and that the client is assigned
        self.assertEqual(next(results).id, 5)

        # test that there are still 9 items available using the list function
        all_results = list(results)
        self.assertEqual(len(all_results), 9)

        # tests the StopIteration exception
        self.assertRaises(StopIteration, next, results)
