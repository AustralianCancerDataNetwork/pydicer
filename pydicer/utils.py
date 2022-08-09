import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pydicom

from pydicer.config import PyDicerConfig


def hash_uid(uid, truncate=6):
    """Hash a UID and truncate it

    Args:
        uid (str): The UID to hash
        truncate (int, optional): The number of the leading characters to keep. Defaults to 6.

    Returns:
        str: The hashed and trucated UID
    """

    hash_sha = hashlib.sha256()
    hash_sha.update(uid.encode("UTF-8"))
    return hash_sha.hexdigest()[:truncate]


def determine_dcm_datetime(ds, require_time=False):
    """Get a date/time value from a DICOM dataset. Will attempt to pull from SeriesDate/SeriesTime
    field first. Will fallback to StudyDate/StudyTime or InstanceCreationDate/InstanceCreationTime
    if not available.

    Args:
        ds (pydicom.Dataset): DICOM dataset
        require_time (bool): Flag to require the time component along with the date

    Returns:
        datetime: The date/time
    """

    date_type_preference = ["Series", "Study", "InstanceCreation"]

    for date_type in date_type_preference:

        type_date = f"{date_type}Date"
        type_time = f"{date_type}Time"
        if type_date in ds and len(ds[type_date].value) > 0:

            if type_time in ds and len(ds[type_time].value) > 0:
                date_time_str = f"{ds[type_date].value}{ds[type_time].value}"
                if "." in date_time_str:
                    return datetime.strptime(date_time_str, "%Y%m%d%H%M%S.%f")

                return datetime.strptime(date_time_str, "%Y%m%d%H%M%S")

            if require_time:
                continue

            return datetime.strptime(ds[type_date].value, "%Y%m%d")

    return None


def load_object_metadata(row):
    """Loads the object's metadata

    Args:
        row (pd.Series): The row of the converted DataFrame for which to load the metadata

    Returns:
        pydicom.Dataset: The dataset object containing the original DICOM metadata
    """

    config = PyDicerConfig()
    metadata_path = Path(config.get_working_dir(), row.path).joinpath("metadata.json")
    with open(metadata_path, "r", encoding="utf8") as json_file:
        ds_dict = json.load(json_file)

    return pydicom.Dataset.from_json(ds_dict, bulk_data_uri_handler=lambda _: None)

def read_preprocessed_data(working_directory: Path):
    """Reads the pydicer preprocessed data

    Args:
        working_directory (Path): Working directory for project

    Raises:
        SystemError: Error raised when preprocessed data doesn't yet exist

    Returns:
        pd.DataFrame: The preprocessed data
    """

    pydicer_directory = working_directory.joinpath(".pydicer")
    preprocessed_file = pydicer_directory.joinpath("preprocessed.csv")
    if not preprocessed_file.exists():
        raise SystemError("Preprocessed data not found, run preprocess step first")

    # Read the csv
    df_preprocess = pd.read_csv(preprocessed_file, index_col=0)

    # Make sure patient id is a string
    df_preprocess.patient_id = df_preprocess.patient_id.astype(str)

    return df_preprocess

def parse_patient_kwarg(patient, dataset_directory):
    """Helper function to prepare patient list from kwarg used in functions throughout pydicer.

    Args:
        patient (list|str): The patient ID or list of patient IDs. If None, all patients in
          dataset_directory are returned.
        dataset_directory (pathlib.Path): The path of the output directory or dataset directory
          (containing the patient folders)

    Raises:
        ValueError: All patient IDs in list aren't of type `str`
        ValueError: patient was not list, str or None.

    Returns:
        list: The list of patient IDs to process
    """

    if isinstance(patient, list):
        if not all(isinstance(x, str) for x in patient):
            raise ValueError("All patient IDs must be of type 'str'")
    elif patient is None:
        patient = [p.name for p in dataset_directory.glob("*") if p.is_dir()]
    else:

        if not isinstance(patient, str) and patient is not None:
            raise ValueError(
                "Patient ID must be list or str. None is a valid to process all patients"
            )
        patient = [patient]

    return patient
