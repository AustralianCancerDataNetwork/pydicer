# pylint: disable=redefined-outer-name,missing-function-docstring

import tempfile
from pathlib import Path
import numpy as np

import pytest

from pydicer import PyDicer
from pydicer.input.test import TestInput


@pytest.fixture
def test_data():
    """Fixture to grab the test data"""

    directory = Path("./testdata")
    directory.mkdir(exist_ok=True, parents=True)

    dicom_directory = directory.joinpath("dicom")
    dicom_directory.mkdir(exist_ok=True, parents=True)

    test_input = TestInput(dicom_directory)
    test_input.fetch_data()

    return directory


def test_pipeline(test_data):
    """End-to-end test of the entire pipeline"""

    with tempfile.TemporaryDirectory() as directory:

        directory = Path(directory)

        dicom_directory = directory.joinpath("dicom")
        dicom_directory.symlink_to(test_data.absolute(), target_is_directory=True)

        pydicer = PyDicer(directory)
        pydicer.add_input(dicom_directory)

        # Preprocess the data fetch to prepare it for conversion
        pydicer.preprocess()

        # Convert the data into the output directory
        pydicer.convert.convert(patient="HNSCC-01-0199")

        # Visualise the converted data
        pydicer.visualise.visualise(patient="HNSCC-01-0199")

        # Dataset selection and preparation
        pydicer.dataset.prepare("clean", "rt_latest_dose")

        # Analysis computing Radiomics and DVH
        pydicer.analyse.compute_radiomics("clean")
        df_rad = pydicer.analyse.get_all_computed_radiomics_for_dataset()

        # Do some spot checks on the radiomics computed for the dataset to confirm the end-to-end
        # test worked
        assert np.isclose(
            (
                df_rad.loc[
                    (df_rad.Contour == "Cord") & (df_rad.Patient == "HNSCC-01-0199"),
                    "firstorder|Energy",
                ].iloc[0]
            ),
            18025962.0,
        )

        assert np.isclose(
            (
                df_rad.loc[
                    (df_rad.Contour == "post_neck") & (df_rad.Patient == "HNSCC-01-0199"),
                    "firstorder|Median",
                ].iloc[0]
            ),
            45.0,
        )

        assert np.isclose(
            (
                df_rad.loc[
                    (df_rad.Contour == "PTV_63_Gy") & (df_rad.Patient == "HNSCC-01-0199"),
                    "firstorder|Skewness",
                ].iloc[0]
            ),
            -0.0053863391917069,
        )

        pydicer.analyse.compute_dvh()
        df_dose_metrics = pydicer.analyse.compute_dose_metrics(
            d_point=[50, 95, 99], v_point=[1, 10], d_cc_point=[1, 5, 10]
        )

        assert np.isclose(
            (df_dose_metrics.loc[df_dose_metrics.label == "Brainstem", "V10"].iloc[0]),
            29.68311309814453,
        )

        assert np.isclose(
            (df_dose_metrics.loc[df_dose_metrics.label == "PTV_57_Gy", "cc"].iloc[0]),
            145.16115188598633,
        )

        assert np.isclose(
            (df_dose_metrics.loc[df_dose_metrics.label == "Lt_Parotid", "D95"].iloc[0]),
            8.3,
        )

        assert np.isclose(
            (df_dose_metrics.loc[df_dose_metrics.label == "GTV", "D99"].iloc[0]),
            70.2,
        )

        assert np.isclose(
            (df_dose_metrics.loc[df_dose_metrics.label == "Rt_Parotid", "D5cc"].iloc[0]),
            70.4,
        )
