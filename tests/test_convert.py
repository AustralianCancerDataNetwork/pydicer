# pylint: disable=redefined-outer-name

import tempfile
from pathlib import Path

import pytest

import SimpleITK as sitk

from pydicer.input.test import TestInput

from pydicer.convert.rtstruct import convert_rtstruct


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
        assert sitk.GetArrayFromImage(brainstem).sum() == 11533
