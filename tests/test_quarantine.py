import os
from pathlib import Path
import pytest
import shutil
import tempfile

import numpy as np
import pandas as pd

from pydicer.quarantine import copy_file_to_quarantine, read_quarantined_data


def test_copy_file_to_quarantine():
    """Test that copy_file_to_quarantine correctly copies the file to quarantine and
    writes the summary entry.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a dummy DICOM file
        dummy_file = tmpdir_path / "dummy.dcm"
        dummy_file.write_text("Some dummy content")

        # Invoke the function
        copy_file_to_quarantine(dummy_file, tmpdir_path, "Test error message")

        quarantine_dir = tmpdir_path / "quarantine"
        summary_file = quarantine_dir / "summary.csv"

        # Assert quarantine directory was created
        assert quarantine_dir.is_dir(), "Quarantine directory was not created."

        # Assert summary file was created
        assert summary_file.exists(), "Summary CSV file was not created."

        # Read the summary CSV
        df_summary = pd.read_csv(summary_file, index_col=0)

        # Check that exactly one entry is in the summary
        assert len(
            df_summary) == 1, "There should be exactly one entry in the summary."

        # Check the summary row
        row = df_summary.iloc[0]
        assert row["error"] == "Test error message"
        assert "file" in row, "'file' column is missing in the summary DataFrame."
        assert "PatientID" in row, "'PatientID' column is missing in the summary DataFrame."

        # Because this is not a valid DICOM, the code defaults PatientID to UNKNOWN
        assert pd.isna(row["PatientID"])

        # The quarantined file is placed under: quarantine_dir / "UNKNOWN" / <parent_folder> / dummy.dcm
        quarantined_file_path = quarantine_dir.joinpath(
            "UNKNOWN", dummy_file.parent.name, dummy_file.name)
        assert quarantined_file_path.exists(
        ), "Quarantined file was not copied to the correct location."


def test_read_quarantined_data():
    """Test that read_quarantined_data reads data from an existing quarantine summary CSV."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        quarantine_dir = tmpdir_path / "quarantine"
        quarantine_dir.mkdir(exist_ok=True)

        # Create a fake summary CSV
        summary_file = quarantine_dir / "summary.csv"
        df_expected = pd.DataFrame([
            {
                "file": "somefile.dcm",
                "error": "some_error",
                "quarantine_dttm": "2024-01-01 00:00:00",
                "PatientID": "UNKNOWN",
                "Modality": None,
                "SOPInstanceUID": None,
                "SeriesDescription": None,
            }
        ])
        df_expected.to_csv(summary_file)

        # Use the function to read it
        df_summary = read_quarantined_data(tmpdir_path)

        # Assert the DataFrame matches our expectations
        assert len(df_summary) == 1, "Expected one record in the summary."
        row = df_summary.iloc[0]
        assert row["file"] == "somefile.dcm"
        assert row["error"] == "some_error"
        assert row["PatientID"] == "UNKNOWN"


def test_read_quarantined_data_no_summary():
    """Test that reading quarantine data raises an error or fails gracefully if summary.csv is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        quarantine_dir = tmpdir_path / "quarantine"
        quarantine_dir.mkdir(exist_ok=True)

        # summary.csv does NOT exist here
        with pytest.raises(FileNotFoundError):
            read_quarantined_data(tmpdir_path)
