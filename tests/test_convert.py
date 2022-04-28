# pylint: disable=redefined-outer-name,missing-function-docstring

import tempfile
import json
from pathlib import Path
import numpy as np

import pytest

import SimpleITK as sitk
import pydicom

from pydicer.input.test import TestInput
from pydicer.input.web import WebInput

from pydicer.convert.headers import convert_dicom_headers
from pydicer.convert.rtstruct import convert_rtstruct
from pydicer.convert.pt import convert_dicom_to_nifti_pt


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


@pytest.fixture
def test_data_all():
    """Fixture to grab the test data with more modalities"""

    directory = Path("./testdata")
    directory.mkdir(exist_ok=True, parents=True)

    working_directory = directory.joinpath("working2")
    working_directory.mkdir(exist_ok=True, parents=True)

    data_url = "http://shorturl.at/acvNQ"
    web_input = WebInput(data_url, working_directory)
    web_input.fetch_data()

    return working_directory


def test_convert_rt_struct(test_data):

    img_files = [
        str(f)
        for f in test_data.joinpath(
            "HNSCC", "HNSCC-01-0199", "10-26-2002-NA-RT SIMULATION-18560", "3.000000-NA-58373"
        ).glob("*.dcm")
    ]

    img_files.sort()

    rt_struct_file = test_data.joinpath(
        "HNSCC",
        "HNSCC-01-0199",
        "10-26-2002-NA-RT SIMULATION-18560",
        "1.000000-NA-59395",
        "1-1.dcm",
    )

    with tempfile.TemporaryDirectory() as output_dir:

        output_path = Path(output_dir)

        convert_rtstruct(
            img_files,
            rt_struct_file,
            prefix="",
            output_dir=output_path,
            output_img=None,
            spacing=None,
        )

        # Make sure there are the correct number of structures
        assert len(list(output_path.glob("*"))) == 38

        # Open a random structure and check that it is correct
        brainstem_path = output_path.joinpath("Brainstem.nii.gz")
        brainstem = sitk.ReadImage(str(brainstem_path))

        assert brainstem.GetSize() == (512, 512, 174)
        assert brainstem.GetSpacing() == (0.9765625, 0.9765625, 2.5)
        assert sitk.GetArrayFromImage(brainstem).sum() == 12450


def test_convert_pet(test_data_all):

    pet_dir = test_data_all.joinpath(
        "1.3.6.1.4.1.14519.5.2.1.1706.8040.995469920533091641707578194770"
    )

    pet_files = [str(f) for f in pet_dir.glob("*.dcm")]

    with tempfile.TemporaryDirectory() as output_dir:

        output_path = Path(output_dir)
        output_file = output_path.joinpath("pet.nii.gz")
        convert_dicom_to_nifti_pt(pet_files, str(output_file))

        assert output_file.exists()

        pet_img = sitk.ReadImage(str(output_file))
        assert pet_img.GetSize() == (128, 128, 91)

        pet_arr = sitk.GetArrayFromImage(pet_img)
        assert np.allclose(pet_arr.max(), 11.9479, atol=0.001)


def test_save_dicom_headers(test_data_all):

    dicom_dir = test_data_all.joinpath(
        "1.3.6.1.4.1.14519.5.2.1.1706.8040.995469920533091641707578194770"
    )

    dicom_file = [str(f) for f in dicom_dir.glob("*.dcm")][0]

    with tempfile.TemporaryDirectory() as output_dir:

        # Save off the headers for this file
        output_path = Path(output_dir)
        output_file = output_path.joinpath("test.json")
        convert_dicom_headers(dicom_file, "", output_file)

        # Check that we can read them again from the JSON
        with open(output_file, "r", encoding="utf8") as json_file:
            ds_dict = json.load(json_file)

        loaded_ds = pydicom.Dataset.from_json(ds_dict, bulk_data_uri_handler=lambda _: None)
        original_ds = pydicom.read_file(dicom_file)

        # Check that some key header values are the same in the original DICOM and the one loaded
        # from JSON
        assert loaded_ds.SeriesInstanceUID == original_ds.SeriesInstanceUID
        assert loaded_ds.PatientID == original_ds.PatientID
        assert loaded_ds.Modality == original_ds.Modality
        assert loaded_ds.SeriesDate == original_ds.SeriesDate
        assert loaded_ds.FrameOfReferenceUID == original_ds.FrameOfReferenceUID
