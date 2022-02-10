# pylint: disable=redefined-outer-name,missing-function-docstring

import tempfile
from pathlib import Path
import numpy as np

import pytest

from pydicer.input.test import TestInput
from pydicer.preprocess.data import PreprocessData
from pydicer.convert.data import ConvertData
from pydicer.visualise.data import VisualiseData
from pydicer.dataset.preparation import PrepareDataset
from pydicer.analyse.data import AnalyseData


@pytest.fixture
def test_data():
    """Fixture to grab the test data"""

    directory = Path("./testdata")
    directory.mkdir(exist_ok=True, parents=True)

    working_directory = directory.joinpath("working")
    working_directory.mkdir(exist_ok=True, parents=True)

    test_input = TestInput(working_directory)
    test_input.fetch_data()

    return working_directory


def test_pipeline(test_data):
    """End-to-end test of the entire pipeline"""

    with tempfile.TemporaryDirectory() as directory:

        directory = Path(directory)

        output_directory = directory.joinpath("output")
        output_directory.mkdir(exist_ok=True, parents=True)

        # Preprocess the data fetch to prepare it for conversion
        preprocessed_data = PreprocessData(test_data, output_directory)
        preprocessed_result = preprocessed_data.preprocess()

        # Convert the data into the output directory
        convert_data = ConvertData(preprocessed_result, output_directory=output_directory)
        convert_data.convert()

        # Visualise the converted data
        visualise_data = VisualiseData(output_directory)
        visualise_data.visualise()

        # Dataset selection and preparation
        prepare_dataset = PrepareDataset(directory)
        prepare_dataset.prepare("clean", "rt_latest_struct")

        # Analysis computing Radiomics and DVH
        analyse = AnalyseData(directory, "clean")
        analyse.compute_radiomics()
        df = analyse.get_all_computed_radiomics_for_dataset()

        # Do some spot checks on the radiomics computed for the dataset to confirm the end-to-end
        # test worked
        assert np.isclose(
            (
                df.loc[
                    (df.Contour == "Brain") & (df.Patient == "HNSCC-01-0019"), "firstorder|Energy"
                ].iloc[0]
            ),
            503558387.0,
        )

        assert np.isclose(
            (
                df.loc[
                    (df.Contour == "post_neck") & (df.Patient == "HNSCC-01-0199"),
                    "firstorder|Median",
                ].iloc[0]
            ),
            45.0,
        )

        assert np.isclose(
            (
                df.loc[
                    (df.Contour == "CTV63") & (df.Patient == "HNSCC-01-0019"),
                    "firstorder|Skewness",
                ].iloc[0]
            ),
            0.4697397883056055,
        )

        assert np.isclose(
            (
                df.loc[
                    (df.Contour == "ptv57") & (df.Patient == "HNSCC-01-0176"),
                    "firstorder|Variance",
                ].iloc[0]
            ),
            32857.812370113126,
        )

        assert np.isclose(
            (
                df.loc[
                    (df.Contour == "brainstem") & (df.Patient == "HNSCC-01-0176"),
                    "firstorder|MeanAbsoluteDeviation",
                ].iloc[0]
            ),
            7.591863524408501,
        )
