from pathlib import Path
import shutil
import datetime

import pandas as pd
import pydicom

# Attempt to store the following meta data keys in the quarantine summary DataFrame
QUARATINE_DICOM_KEYS = ["PatientID", "Modality", "SOPInstanceUID", "SeriesDescription"]


def copy_file_to_quarantine(file, working_directory, error_msg):
    """Move a DICOM file that couldn't be processed into the quarantine directory

    Args:
        file (Path): DICOM path to be moved into quarantine
        working_directory (pathlib.Path): Main working directory for pydicer
        error_msg (str): error message associated with the quarantined file
    """

    # Attempt to get some header information from the DICOM object to write into the summary

    summary_dict = {"file": file, "error": error_msg, "quarantine_dttm": datetime.datetime.now()}

    ds = pydicom.read_file(file, force=True)
    for k in QUARATINE_DICOM_KEYS:

        val = None
        if k in ds:
            val = ds[k].value

        summary_dict[k] = val

    pat_id = "UNKNOWN"
    if "PatientID" in ds:
        pat_id = ds.PatientID

    df_this_summary = pd.DataFrame([summary_dict])

    quaran_dir = Path(working_directory).joinpath("quarantine")
    file_dir = quaran_dir.joinpath(pat_id, file.parent.name)
    summary_file = quaran_dir.joinpath("summary.csv")

    df_summary = None
    if summary_file.exists():
        df_summary = pd.read_csv(summary_file, index_col=0)
        df_summary = pd.concat([df_summary, df_this_summary], ignore_index=True)
    else:
        df_summary = df_this_summary

    # Create "quarantine/PATH_TO_DCM" directory
    file_dir.mkdir(exist_ok=True, parents=True)

    # Copy original DCM file to quarantine area
    shutil.copyfile(file, file_dir.joinpath(file.name))

    # Create (if doesn't exist) summary file to hold info about file error
    df_summary.to_csv(summary_file)


class TreatImages:
    """
    Class to treat the quarantined images and prepare it for further processing

    Args:
        quaran_directory (Path): path to the quarantine directory
    """

    def __init__(self, quaran_directory):
        self.quaran_directory = quaran_directory
