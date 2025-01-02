import os
from pathlib import Path
import tempfile

import pytest
from unittest.mock import MagicMock, patch

from pydicer.input.web import WebInput
from pydicer.input.test import TestInput
from pydicer.input.filesystem import FileSystemInput
from pydicer.input.pacs import DICOMPACSInput
from pydicer.input.tcia import TCIAInput


def test_input_valid_working_dir():
    valid_test_input = WebInput(data_url="")
    # Assert path to DICOMs exists
    assert valid_test_input.working_directory.is_dir()

    valid_filesystem_input = FileSystemInput(
        valid_test_input.working_directory)
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
                if os.path.isfile(os.path.join(invalid_tcia_input.working_directory, name))
            ]
        )
        == 0
    )


def test_input_invalid_working_dir():
    invalid_test_input = WebInput(
        data_url="", working_directory="INVALID_PATH")
    # Assert path to DICOMs does not exist
    assert not invalid_test_input.working_directory.is_dir()

    with pytest.raises(FileNotFoundError):
        FileSystemInput("INVALID_PATH")

    invalid_work_dir_tcia_input = TCIAInput(
        collection="TCGA-GBM",
        patient_ids=["TCGA-08-0244"],
        modalities=["MR"],
        working_directory="INVALID_PATH",
    )
    # Assert path to DICOMs does not exist
    assert not invalid_work_dir_tcia_input.working_directory.is_dir()


@pytest.mark.skip
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


def test_dicom_pacs_invalid_host():
    # Using a presumably incorrect host/port to force a ConnectionError
    with pytest.raises(ConnectionError):
        DICOMPACSInput("INCORRECT_HOST", 1234)


def test_dicom_pacs_valid_host(mocker):
    """
    Test creating a DICOMPACSInput instance with a valid host where verify() returns True.
    """
    # Patch the DicomConnector to return True for verify()
    mock_connector_class = mocker.patch("pydicer.input.pacs.DicomConnector")
    mock_connector_instance = mock_connector_class.return_value
    mock_connector_instance.verify.return_value = True

    # Should not raise ConnectionError
    dicompacs_input = DICOMPACSInput("VALID_HOST", 11112, "AE_TITLE")

    # Assert the underlying connector was indeed created
    assert dicompacs_input.dicom_connector is not None
    assert dicompacs_input.working_directory.is_dir()
    # Verify that verify() was called exactly once on initialization
    mock_connector_instance.verify.assert_called_once()


def test_dicom_pacs_fetch_data_success(mocker):
    """
    Test fetching data when the connection is valid, ensuring that we:
    1) Convert single string patients/modalities to lists
    2) Skip 'None' returns from do_find
    3) Skip series whose patient ID doesn't match
    4) Renames downloaded files to .dcm
    """
    mock_connector_class = mocker.patch("pydicer.input.pacs.DicomConnector")
    mock_connector_instance = mock_connector_class.return_value
    mock_connector_instance.verify.return_value = True

    # Mock do_find to return "studies" and then "series"
    # The top-level do_find returns a list of "study" datasets (some None),
    # then the second do_find returns a list of "series" datasets (some None).
    # Each dataset is just a MagicMock or simple object with needed attributes.
    mock_study_1 = MagicMock()
    mock_study_1.StudyInstanceUID = "STUDY_UID_1"
    mock_study_2 = MagicMock()
    mock_study_2.StudyInstanceUID = "STUDY_UID_2"
    mock_study_none = None  # Should be skipped

    mock_series_1 = MagicMock()
    mock_series_1.SeriesInstanceUID = "SERIES_UID_1"
    mock_series_1.PatientID = "PATIENT_1"
    mock_series_2 = MagicMock()
    mock_series_2.SeriesInstanceUID = "SERIES_UID_2"
    mock_series_2.PatientID = "SOME_OTHER_PATIENT"  # Should be skipped
    mock_series_none = None

    # The "find" calls for studies:
    mock_connector_instance.do_find.side_effect = [
        [mock_study_1, mock_study_none, mock_study_2],  # Studies
        [mock_series_1, mock_series_none, mock_series_2],  # Series for STUDY_UID_1
        # Series for STUDY_UID_2 (just re-using same object to keep it simple)
        [mock_series_1],
    ]

    dicompacs_input = DICOMPACSInput("VALID_HOST", 11112, "AE_TITLE")

    # Create a dummy file that doesn't end with .dcm so we can test rename
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        # Force working directory to our temp dir
        dicompacs_input.working_directory = tmpdir_path

        dummy_file_path = tmpdir_path / "dummy_no_ext"
        dummy_file_path.write_text("test file content")

        # Single patient, single modality as strings
        dicompacs_input.fetch_data("PATIENT_1", "CT")

        # Ensure do_find was called multiple times
        # The first call: study-level (QueryRetrieveLevel="STUDY")
        # The next calls: series-level (QueryRetrieveLevel="SERIES"), once for each study
        assert mock_connector_instance.do_find.call_count == 3

        # Ensure download_series was called for the valid series only
        # The second series had a mismatched patient ID, so skip
        # The third call is a new do_find -> leads to another series (mock_series_1 with same patient)
        # So we should have downloaded 2 times
        assert mock_connector_instance.download_series.call_count == 2
        call_args_list = mock_connector_instance.download_series.call_args_list
        # We expect the arguments to match "SERIES_UID_1" each time in this example
        # (in practice, could differ if you had different series objects)
        assert call_args_list[0][0][0] == "SERIES_UID_1"
        assert call_args_list[1][0][0] == "SERIES_UID_1"

        # Check that the file without extension was renamed to .dcm
        renamed_file = tmpdir_path / "dummy_no_ext.dcm"
        assert renamed_file.exists(), "File without .dcm extension should have been renamed."
        assert not dummy_file_path.exists(), "Original file without extension should be renamed."


@pytest.mark.skip
def test_dicom_pacs_fetch():
    """
    Example real test that tries to actually fetch from a public DICOM PACS.
    This might be skipped because it depends on external availability.
    """
    pacs_input = DICOMPACSInput("www.dicomserver.co.uk", 11112, "DCMQUERY")
    pacs_input.fetch_data("PAT004", modalities=["GM"])

    assert pacs_input.working_directory.is_dir()
    assert len(list(pacs_input.working_directory.glob("*/*"))) > 0
