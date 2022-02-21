import hashlib
import json
from datetime import datetime
from pathlib import Path

import pydicom


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


def determine_dcm_datetime(ds):
    """Get a date/time value from a DICOM dataset. Will attempt to pull from SeriesDate/SeriesTime
    field first. Will fallback to StudyDate/StudyTime or InstanceCreationDate/InstanceCreationTime
    if not available.

    Args:
        ds (pydicom.Dataset): DICOM dataset

    Returns:
        datetime: The date/time
    """

    if "SeriesDate" in ds and len(ds.SeriesDate) > 0:

        if "SeriesTime" in ds and len(ds.SeriesTime) > 0:
            date_time_str = f"{ds.SeriesDate}{ds.SeriesTime}"
            if "." in date_time_str:
                return datetime.strptime(date_time_str, "%Y%m%d%H%M%S.%f")

            return datetime.strptime(date_time_str, "%Y%m%d%H%M%S")

        return datetime.strptime(ds.SeriesDate, "%Y%m%d")

    if "StudyDate" in ds and len(ds.StudyDate) > 0:

        if "StudyTime" in ds and len(ds.StudyTime) > 0:
            date_time_str = f"{ds.StudyDate}{ds.StudyTime}"
            if "." in date_time_str:
                return datetime.strptime(date_time_str, "%Y%m%d%H%M%S.%f")

            return datetime.strptime(date_time_str, "%Y%m%d%H%M%S")

        return datetime.strptime(ds.StudyDate, "%Y%m%d")

    if "InstanceCreationDate" in ds and len(ds.InstanceCreationDate) > 0:

        if "InstanceCreationTime" in ds and len(ds.InstanceCreationTime) > 0:
            date_time_str = f"{ds.InstanceCreationDate}{ds.InstanceCreationTime}"
            if "." in date_time_str:
                return datetime.strptime(date_time_str, "%Y%m%d%H%M%S.%f")

            return datetime.strptime(date_time_str, "%Y%m%d%H%M%S")

        return datetime.strptime(ds.InstanceCreationDate, "%Y%m%d")

    return None


def load_object_metadata(row):
    """Loads the object's metadata

    Args:
        row (pd.Series): The row of the converted DataFrame for which to load the metadata

    Returns:
        pydicom.Dataset: The dataset object containing the original DICOM metadata
    """

    metadata_path = Path(row.path).joinpath("metadata.json")
    with open(metadata_path, "r", encoding="utf8") as json_file:
        ds_dict = json.load(json_file)

    return pydicom.Dataset.from_json(ds_dict, bulk_data_uri_handler=lambda _: None)
