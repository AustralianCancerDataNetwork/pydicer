import os
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
import pydicom

from pydicer.constants import CONVERTED_DIR_NAME, PYDICER_DIR_NAME

logger = logging.getLogger(__name__)


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

    metadata_path = Path(row.path).joinpath("metadata.json")

    if not metadata_path.exists():
        return pydicom.Dataset()
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

    pydicer_directory = working_directory.joinpath(PYDICER_DIR_NAME)
    preprocessed_file = pydicer_directory.joinpath("preprocessed.csv")
    if not preprocessed_file.exists():
        raise SystemError("Preprocessed data not found, run preprocess step first")

    # Read the csv
    df_preprocess = pd.read_csv(preprocessed_file, index_col=0)

    # Make sure patient id is a string
    df_preprocess.patient_id = df_preprocess.patient_id.astype(str)

    return df_preprocess


def read_converted_data(
    working_directory: Path,
    dataset_name=CONVERTED_DIR_NAME,
    patients=None,
    join_working_directory=True,
):
    """Read the converted data frame from the supplied data directory.

    Args:
        working_directory (Path): Working directory for project
        dataset_name (str, optional): The name of the dataset for which to extract converted data.
          Defaults to "data".
        patients (list, optional): The list of patients for which to read converted data. If None
            is supplied then all data will be read. Defaults to None.
        join_working_directory (bool, optional): If True, the path in the data frame returned will
            be adjusted to the location of the working_directory. If False the path will be
            relative to the working_directory.

    Returns:
        pd.DataFrame: The DataFrame with the converted data objects.
    """

    dataset_directory = working_directory.joinpath(dataset_name)

    if not dataset_directory.exists():
        raise SystemError(f"Dataset directory {dataset_directory} does not exist")

    df = pd.DataFrame()

    for pat_dir in dataset_directory.glob("*"):

        if not pat_dir.is_dir():
            continue

        pat_id = pat_dir.name

        if patients is not None:
            if pat_id not in patients:
                continue

        # Read in the DataFrame storing the converted data for this patient
        converted_csv = dataset_directory.joinpath(pat_id, "converted.csv")
        if not converted_csv.exists():
            logger.warning("Converted CSV doesn't exist for %s", pat_id)
            continue

        df_converted = pd.read_csv(converted_csv, index_col=0)
        df = pd.concat([df, df_converted])

    # Make sure patient id is a string
    df.patient_id = df.patient_id.astype(str)

    # Join the working directory to each object's path
    if join_working_directory:
        df.path = df.path.apply(lambda p: os.path.join(working_directory, p))

    return df.reset_index(drop=True)


def parse_patient_kwarg(patient):
    """Helper function to prepare patient list from kwarg used in functions throughout pydicer.

    Args:
        patient (list|str): The patient ID or list of patient IDs. If None, all patients in
          dataset_directory are returned.

    Raises:
        ValueError: All patient IDs in list aren't of type `str`
        ValueError: patient was not list, str or None.

    Returns:
        list: The list of patient IDs to process or None if patient is None
    """

    if isinstance(patient, list):
        if not all(isinstance(x, str) for x in patient):
            raise ValueError("All patient IDs must be of type 'str'")
    elif patient is None:
        return None
    else:
        if not isinstance(patient, str) and patient is not None:
            raise ValueError(
                "Patient ID must be list or str. None is a valid to process all patients"
            )
        patient = [patient]

    return patient


def read_simple_itk_image(row):

    object_path = Path(row.path)

    nifti_path = object_path.joinpath(f"{row.modality}.nii.gz")

    if not nifti_path.exists():
        logger.warning("Unable to load Nifti at %s", nifti_path)
        return None

    return sitk.ReadImage(str(nifti_path))
