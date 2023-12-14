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
from dhl_sdk.exceptions import (
    InvalidSpectraException,
    InvalidInputsException,
    InvalidTimestampsException,
)
from dhl_sdk._input_processing import (
    format_predictions,
    _validate_spectra_format,
    Preprocessor,
    SpectraPreprocessor,
    CultivationPreprocessor,
)


class TestSpectraUtils(unittest.TestCase):
    def setUp(self):
        spectrum_var = {
            "id": "ram-111",
            "code": "spc1",
            "variant": "spectrum",
            "name": "raman 1",
            "spectrum": {"xAxis": {"dimension": 4}},
        }
        self.model_no_inputs = Mock()
        self.model_no_inputs.inputs = []
        self.model_with_inputs = Mock()
        self.model_with_inputs.dataset.variables = [
            Variable(**spectrum_var),
            Variable(id="id-123", code="var1", variant="numeric", name="variable 1"),
            Variable(id="id-456", code="var2", variant="numeric", name="variable 2"),
            Variable(id="id-789", code="out1", variant="numeric", name="output 1"),
            Variable(id="id-101", code="out2", variant="numeric", name="output 2"),
        ]
        self.model_with_inputs.inputs = ["id-123", "id-456"]

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

    def test_validation_with_input(self):
        model = self.model_with_inputs
        model.spectra_size = 4

        spectra = [[1, 2, 3, 3], [4, 5, 6, 6], [7, 8, 9, 9]]
        processor = Preprocessor(
            SpectraPreprocessor(spectra=spectra, model=model, inputs=None)
        )
        self.assertRaises(
            InvalidInputsException,
            processor.validate,
        )

        inputs = {"var10": [0, 1, 0], "var2": [1, 1, 1]}
        processor = Preprocessor(
            SpectraPreprocessor(spectra=spectra, model=model, inputs=inputs)
        )
        with self.assertRaises(InvalidInputsException) as ex:
            processor.convert_to_request()
            self.assertTrue(
                ex.exception.message.startswith(
                    "No matching Input found for key: var10"
                )
            )

        inputs = {"var1": [0, 1, 0], "var2": [1, 1, 1]}
        processor = Preprocessor(
            SpectraPreprocessor(spectra=spectra, model=model, inputs=inputs)
        )
        processor.validate()

    def test_convert_to_request(self):
        model = self.model_with_inputs
        model.spectra_size = 4
        model.dataset.get_spectrum_index.return_value = 0

        spectra = [[1, 2, 3, 3], [4, 5, 6, 6], [7, 8, 9, 9]]
        inputs = {"var1": [0, 1, 0], "var2": [1, 1, 1]}
        processor = Preprocessor(
            SpectraPreprocessor(spectra=spectra, model=model, inputs=inputs)
        )

        request = processor.convert_to_request()

        self.assertEqual(request[0]["instances"][0][0]["sampleId"], ["0", "1", "2"])
        self.assertEqual(request[0]["instances"][0][0]["values"][0], [1, 2, 3, 3])
        self.assertEqual(request[0]["instances"][0][0]["values"][1], [4, 5, 6, 6])
        self.assertEqual(request[0]["instances"][0][0]["values"][2], [7, 8, 9, 9])
        self.assertEqual(request[0]["instances"][0][2]["values"], [1, 1, 1])


class TestCultivationUtils(unittest.TestCase):
    def setUp(self):
        var1 = {
            "id": "id-123",
            "code": "var1",
            "variant": "numeric",
            "name": "variable 1",
            "group": {"code": "X"},
        }
        var2 = {
            "id": "id-456",
            "code": "var2",
            "variant": "numeric",
            "name": "variable 2",
            "group": {"code": "X"},
        }
        var3 = {
            "id": "id-789",
            "code": "var3",
            "variant": "numeric",
            "name": "output 1",
            "group": {"code": "W"},
        }
        var4 = {
            "id": "id-101",
            "code": "var4",
            "variant": "numeric",
            "name": "output 2",
            "group": {"code": "Y"},
        }
        self.model = Mock()
        self.model.dataset.variables = [
            Variable(**var1),
            Variable(**var2),
            Variable(**var3),
            Variable(**var4),
        ]

    def test_format_predictions(self):
        predictions = [
            PredictResponse(
                instances=[
                    [
                        Instance(values=[1, 2, 3]),
                        Instance(values=[4, 5, 6]),
                        None,
                        None,
                    ]
                ]
            ),
            PredictResponse(
                instances=[
                    [
                        Instance(values=[1, 2, 3]),
                        Instance(values=[4, 5, 6]),
                        None,
                        None,
                    ]
                ]
            ),
        ]

        formatted_predictions = format_predictions(predictions, model=self.model)

        self.assertDictEqual(
            formatted_predictions,
            {"var1": [1, 2, 3, 1, 2, 3], "var2": [4, 5, 6, 4, 5, 6]},
        )

    def test_timestamp_validation(self):
        model = self.model
        inputs = {"var1": [10], "var2": [20], "var3": [1, 2, 3], "var4": [40]}
        timestamps = [1, 2, 3]

        processor = Preprocessor(
            CultivationPreprocessor(timestamps, "s", inputs, model)
        )
        processor.validate()

        processor = Preprocessor(CultivationPreprocessor([1, 2], "s", inputs, model))
        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "The recipe requires var3 to be complete"
                )
            )

        processor = Preprocessor(CultivationPreprocessor([6, 4, 3], "s", inputs, model))
        with self.assertRaises(InvalidTimestampsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith("Timestamps must be in ascending order")
            )

        processor = Preprocessor(CultivationPreprocessor([1, 2, 3], "m", inputs, model))
        processor.validate()
        self.assertEqual(processor._strategy.timestamps, [60, 120, 180])

        processor = Preprocessor(
            CultivationPreprocessor(["1", 2, 3], "h", inputs, model)
        )
        with self.assertRaises(InvalidTimestampsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "All values of timestamps must be valid numeric values"
                )
            )

    def test_input_validation(self):
        model = self.model
        inputs = {"var1": [10], "var2": [20], "var3": [1, 2, 3], "var4": [40]}
        timestamps = [1, 2, 3]

        processor = Preprocessor(
            CultivationPreprocessor(timestamps, "s", inputs, model)
        )
        processor.validate()

        inputs = {"var1": [10], "var3": [1, 2, 3], "var4": [40]}
        processor = Preprocessor(
            CultivationPreprocessor(timestamps, "s", inputs, model)
        )
        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertEqual(
                ex.exception.message,
                "Input var2 is a X Variable, so it must be provided",
            )

        inputs = {"var1": [10], "var2": [20], "var3": [1], "var4": [40]}
        processor = Preprocessor(
            CultivationPreprocessor(timestamps, "s", inputs, model)
        )
        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "The recipe requires var3 to be complete"
                )
            )

        inputs = {"var1": [10], "var2": [20], "var3": [1, 3, 5]}
        processor = Preprocessor(
            CultivationPreprocessor(timestamps, "s", inputs, model)
        )
        processor.validate()

        inputs = {"var1": [10, 20], "var2": [20], "var3": [1, 3, 5]}
        processor = Preprocessor(
            CultivationPreprocessor(timestamps, "s", inputs, model)
        )
        with self.assertRaises(InvalidInputsException) as ex:
            processor.validate()
            self.assertTrue(
                ex.exception.message.startswith(
                    "Input var1 is only requires initial values"
                )
            )

    def test_convert_to_request(self):
        model = self.model
        inputs = {
            "var4": [40],
            "var1": [10],
            "var3": [0.2, 0.6, 0.6, 0.1],
            "var2": [20],
        }
        timestamps = [0, 1, 2, 3]

        processor = Preprocessor(
            CultivationPreprocessor(timestamps, "h", inputs, model)
        )

        request = processor.convert_to_request()

        print()

        self.assertEqual(request[0]["instances"][0][0]["values"], [10])  # var1
        self.assertEqual(request[0]["instances"][0][0]["timestamps"], [0])
        self.assertEqual(request[0]["instances"][0][1]["values"], [20])  # var2
        self.assertEqual(
            request[0]["instances"][0][2]["values"], [0.2, 0.6, 0.6, 0.1]
        )  # var3
        self.assertEqual(
            request[0]["instances"][0][2]["timestamps"], [0, 3600, 7200, 10800]
        )


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
