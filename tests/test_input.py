import pytest

from pydicer.input.web import WebInput, download_and_extract_zip_file
from pydicer.input.test import TestInput
from pydicer.input.filesystem import FilesystemInput


def test_input_valid_working_dir_():
    valid_test_input = WebInput(data_url="")
    # Assert path to DICOMs exists
    assert valid_test_input.working_directory.is_dir()

    valid_filesystem_input = FilesystemInput(valid_test_input.working_directory)
    # Assert path to DICOMs exists
    assert valid_filesystem_input.working_directory.is_dir()


def test_input_invalid_working_dir_():
    invalid_test_input = WebInput(working_directory="NOT_VALID_PATH", data_url="")
    # Assert path to DICOMs does not exist
    assert not invalid_test_input.working_directory.is_dir()

    with pytest.raises(Exception):
        FilesystemInput("NOT_VALID_PATH")


def test_test_input():

    test_input = TestInput()

    download_and_extract_zip_file(test_input.data_url, test_input.working_directory)
    output_directory = test_input.working_directory.joinpath("HNSCC")

    # Assert that the 3 directories now exist on the system filepath
    assert output_directory.joinpath("HNSCC-01-0019").is_dir()
    assert output_directory.joinpath("HNSCC-01-0176").is_dir()
    assert output_directory.joinpath("HNSCC-01-0199").is_dir()
