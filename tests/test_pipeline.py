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

    working_directory = directory.joinpath("dicom")
    working_directory.mkdir(exist_ok=True, parents=True)

    test_input = TestInput(working_directory)
    test_input.fetch_data()

    return working_directory


# def test_pipeline(test_data):
#     """End-to-end test of the entire pipeline"""

#     with tempfile.TemporaryDirectory() as directory:

#         directory = Path(directory)

#         dicom_directory = directory.joinpath("dicom")
#         dicom_directory.symlink_to(test_data.absolute(), target_is_directory=True)

#         # Preprocess the data fetch to prepare it for conversion
#         preprocessed_data = PreprocessData(directory)
#         preprocessed_data.preprocess()

#         # Convert the data into the output directory
#         convert_data = ConvertData(directory)
#         convert_data.convert(patient="HNSCC-01-0199")

#         # Visualise the converted data
#         visualise_data = VisualiseData(directory)
#         visualise_data.visualise()

#         # Dataset selection and preparation
#         prepare_dataset = PrepareDataset(directory)
#         prepare_dataset.prepare("clean", "rt_latest_dose")

#         # Analysis computing Radiomics and DVH
#         analyse = AnalyseData(directory, "clean")
#         analyse.compute_radiomics()
#         df_rad = analyse.get_all_computed_radiomics_for_dataset()

#         # Do some spot checks on the radiomics computed for the dataset to confirm the end-to-end
#         # test worked
#         assert np.isclose(
#             (
#                 df_rad.loc[
#                     (df_rad.Contour == "Cord") & (df_rad.Patient == "HNSCC-01-0199"),
#                     "firstorder|Energy",
#                 ].iloc[0]
#             ),
#             16604633.0,
#         )

#         assert np.isclose(
#             (
#                 df_rad.loc[
#                     (df_rad.Contour == "post_neck") & (df_rad.Patient == "HNSCC-01-0199"),
#                     "firstorder|Median",
#                 ].iloc[0]
#             ),
#             45.0,
#         )

#         assert np.isclose(
#             (
#                 df_rad.loc[
#                     (df_rad.Contour == "PTV_63_Gy") & (df_rad.Patient == "HNSCC-01-0199"),
#                     "firstorder|Skewness",
#                 ].iloc[0]
#             ),
#             0.0914752043992083,
#         )

#         analyse.compute_dvh()
#         df_dose_metrics = analyse.compute_dose_metrics(
#             d_point=[50, 95, 99], v_point=[1, 10], d_cc_point=[1, 5, 10]
#         )

#         assert np.isclose(
#             (df_dose_metrics.loc[df_dose_metrics.label == "Brainstem", "V10"].iloc[0]),
#             27.496815,
#         )

#         assert np.isclose(
#             (df_dose_metrics.loc[df_dose_metrics.label == "PTV_57_Gy", "cc"].iloc[0]),
#             132.961273,
#         )

#         assert np.isclose(
#             (df_dose_metrics.loc[df_dose_metrics.label == "Lt_Parotid", "D95"].iloc[0]),
#             8.4,
#         )

#         assert np.isclose(
#             (df_dose_metrics.loc[df_dose_metrics.label == "GTV", "D99"].iloc[0]),
#             70.2,
#         )

#         assert np.isclose(
#             (df_dose_metrics.loc[df_dose_metrics.label == "Rt_Parotid", "D5cc"].iloc[0]),
#             70.2,
#         )
