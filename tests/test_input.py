import os

from pydicer.input.web import WebInput, download_and_extract_zip_file
from pydicer.input.test import TestInput
from pydicer.input.filesystem import FilesystemInput
from pydicer.input.tcia import TCIAInput


def test_input_valid_working_dir():
    valid_test_input = WebInput(data_url="")
    # Assert path to DICOMs exists
    assert valid_test_input.working_directory.is_dir()

    valid_filesystem_input = FilesystemInput()
    # Assert path to DICOMs exists
    assert valid_filesystem_input.working_directory.is_dir()

    valid_tcia_input = TCIAInput(collection="", patient_ids=[], modalities=[])
    # Assert path to DICOMs exists
    assert valid_tcia_input.working_directory.is_dir()


def assert_invalid_tcia_input(invalid_tcia_input):
    """
    Assert path to DICOMs does exist, but it contains no files
    """
    invalid_tcia_input.fetch_data()
    assert invalid_tcia_input.working_directory.is_dir()
    assert (
        len(
            [
                name
                for name in os.listdir(invalid_tcia_input.working_directory)
                if os.path.isfile(name)
            ]
        )
        == 0
    )


def test_input_invalid_working_dir():
    invalid_test_input = WebInput(data_url="", working_directory="INVALID_PATH")
    # Assert path to DICOMs does not exist
    assert not invalid_test_input.working_directory.is_dir()

    invalid_filesystem_input = FilesystemInput(working_directory="INVALID_PATH")
    # Assert path to DICOMs does not exist
    assert not invalid_filesystem_input.working_directory.is_dir()

    invalid_work_dir_tcia_input = TCIAInput(
        collection="TCGA-GBM",
        patient_ids=["TCGA-08-0244"],
        modalities=["MR"],
        working_directory="INVALID_PATH",
    )
    # Assert path to DICOMs does not exist
    assert not invalid_work_dir_tcia_input.working_directory.is_dir()


def test_tcia_input():
    invalid_collection_tcia_input = TCIAInput(
        collection="INVALID_COLLECTION", patient_ids=[], modalities=[]
    )
    invalid_patient_id_tcia_input = TCIAInput(
        collection="TCGA-GBM", patient_ids=["INVALID_PATIENT_ID"], modalities=[]
    )

    assert_invalid_tcia_input(invalid_collection_tcia_input)
    assert_invalid_tcia_input(invalid_patient_id_tcia_input)


def test_test_input():

    test_input = TestInput()

    test_input.fetch_data()
    output_directory = test_input.working_directory.joinpath("HNSCC")

    # Assert that the 3 directories now exist on the system filepath
    assert output_directory.joinpath("HNSCC-01-0019").is_dir()
    assert output_directory.joinpath("HNSCC-01-0176").is_dir()
    assert output_directory.joinpath("HNSCC-01-0199").is_dir()
