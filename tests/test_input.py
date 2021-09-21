import pytest

from pydicer.input.web import WebInput, download_and_extract_zip_file
from pydicer.input.test import TestInput
from pydicer.input.filesystem import FilesystemInput
from pydicer.input.pacs import DICOMPACSInput


def test_input_valid_working_dir_():
    valid_test_input = WebInput(data_url="")
    # Assert path to DICOMs exists
    assert valid_test_input.working_directory.is_dir()

    valid_filesystem_input = FilesystemInput()
    # Assert path to DICOMs exists
    assert valid_filesystem_input.working_directory.is_dir()


def test_input_invalid_working_dir_():
    invalid_test_input = WebInput(working_directory="NOT_VALID_PATH", data_url="")
    # Assert path to DICOMs does not exist
    assert not invalid_test_input.working_directory.is_dir()

    invalid_filesystem_input = FilesystemInput(working_directory="NOT_VALID_PATH")
    # Assert path to DICOMs does not exist
    assert not invalid_filesystem_input.working_directory.is_dir()


def test_test_input():

    test_input = TestInput()

    download_and_extract_zip_file(test_input.data_url, test_input.working_directory)
    output_directory = test_input.working_directory.joinpath("HNSCC")

    # Assert that the 3 directories now exist on the system filepath
    assert output_directory.joinpath("HNSCC-01-0019").is_dir()
    assert output_directory.joinpath("HNSCC-01-0176").is_dir()
    assert output_directory.joinpath("HNSCC-01-0199").is_dir()


def test_dicom_pacs_invalid_host():

    # Using this public DICOM PACS for testing, not sure how reliable this one is to provide this
    # data consistently
    with pytest.raises(ConnectionError):
        DICOMPACSInput("INCORRECT_HOST", 1234)


def test_dicom_pacs_fetch():

    # Using this public DICOM PACS for testing, not sure how reliable this one is to provide this
    # data consistently. Downloading some General Miroscopy data here, which doesn't really matter
    # because this is just to check that data is being downloaded.
    pacs_input = DICOMPACSInput("www.dicomserver.co.uk", 11112, "DCMQUERY")
    pacs_input.fetch_data("PAT004", modalities=["GM"])

    assert pacs_input.working_directory.is_dir()

    assert len([p for p in pacs_input.working_directory.glob("*/*")]) > 0
