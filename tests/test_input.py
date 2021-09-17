from pydicer.input.web import WebInput, download_and_extract_zip_file
from pydicer.input.test import TestInput
from pydicer.input.filesystem import FilesystemInput

import os

def test_input_valid_working_dir_():
    valid_test_input = WebInput(data_url="")
    # Assert path to DICOMs exists
    assert os.path.isdir(valid_test_input.working_directory)

    valid_filesystem_input = FilesystemInput()
    # Assert path to DICOMs exists
    assert os.path.isdir(valid_filesystem_input.working_directory)

def test_input_invalid_working_dir_():
    invalid_test_input = WebInput(working_directory="NOT_VALID_PATH", data_url="")
    # Assert path to DICOMs does not exist
    assert not os.path.isdir(invalid_test_input.working_directory)

    invalid_filesystem_input = FilesystemInput(working_directory="NOT_VALID_PATH")
    # Assert path to DICOMs does not exist
    assert not os.path.isdir(invalid_filesystem_input.working_directory)

def test_test_input():

    test_input = TestInput()

    download_and_extract_zip_file(test_input.data_url, test_input.working_directory)
    output_directory = test_input.working_directory.joinpath("HNSCC")

    # Assert that the 3 directories now exist on the system filepath
    assert os.path.isdir(output_directory.joinpath("HNSCC-01-0019"))
    assert os.path.isdir(output_directory.joinpath("HNSCC-01-0176"))
    assert os.path.isdir(output_directory.joinpath("HNSCC-01-0199"))
