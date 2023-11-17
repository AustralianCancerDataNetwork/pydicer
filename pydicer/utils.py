import os
import hashlib
import json
import logging
import tempfile
import urllib
import zipfile
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import SimpleITK as sitk
import pydicom
import tqdm

from pydicer.config import PyDicerConfig
from pydicer.constants import CONVERTED_DIR_NAME, PYDICER_DIR_NAME, DEFAULT_MAPPING_ID

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


def load_object_metadata(row: pd.Series, keep_tags=None, remove_tags=None):
    """Loads the object's metadata

    Args:
        row (pd.Series): The row of the converted DataFrame for which to load the metadata
        keep_tags (str|list, optional): DICOM tag keywords keep when loading data. If set all other
          tags will be removed. Defaults to None.
        remove_tag (str|list, optional): DICOM tag keywords keep when loading data. If set all
          other tags will be kept. Defaults to None.

    Returns:
        pydicom.Dataset: The dataset object containing the original DICOM metadata
    """

    row_path = Path(row.path)

    config = PyDicerConfig()

    # If the working directory is configured and the row_path isn't relative to it, join it.
    if config is not None:
        try:
            row_path.relative_to(config.get_working_dir())
        except ValueError:
            row_path = config.get_working_dir().joinpath(row_path)

    metadata_path = row_path.joinpath("metadata.json")

    if not metadata_path.exists():
        return pydicom.Dataset()
    with open(metadata_path, "r", encoding="utf8") as json_file:
        ds_dict = json.load(json_file)

    if keep_tags is not None:
        clean_keep_tags = []

        if not isinstance(keep_tags, list):
            keep_tags = [keep_tags]

        for tag in keep_tags:
            tag_key = pydicom.datadict.tag_for_keyword(tag)
            if tag_key is not None:
                t = pydicom.tag.Tag(tag_key)
                group_str = hex(t.group).replace("0x", "").zfill(4)
                element_str = hex(t.element).replace("0x", "").zfill(4)
                tag = f"{group_str}{element_str}"

            clean_keep_tags.append(tag)

        keep_tags = clean_keep_tags

    # If
    if keep_tags is not None:
        if remove_tags is None:
            remove_tags = []

        for tag in ds_dict.keys():
            if tag.lower() not in keep_tags:
                remove_tags.append(tag)

    if remove_tags is not None:
        if not isinstance(remove_tags, list):
            remove_tags = [remove_tags]

        for tag in remove_tags:
            tag_key = pydicom.datadict.tag_for_keyword(tag)
            if tag_key is not None:
                t = pydicom.tag.Tag(tag_key)
                group_str = hex(t.group).replace("0x", "").zfill(4)
                element_str = hex(t.element).replace("0x", "").zfill(4)
                tag = f"{group_str}{element_str}".upper()

            if tag in ds_dict:
                del ds_dict[tag]

    return pydicom.Dataset.from_json(ds_dict, bulk_data_uri_handler=lambda _: None)


def load_dvh(row, struct_hash=None):
    """Loads an object's Dose Volume Histogram (DVH)

    Args:
        row (pd.Series): The row of the converted DataFrame for an RTDOSE
        struct_hash (list|str, optional): The struct_hash (or list of struct_hashes) to load DVHs
            for. When None all DVHs for RTDOSE will be loaded. Defaults to None.

    Raises:
        ValueError: Raised the the object described in the row is not an RTDOSE

    Returns:
        pd.DataFrame: The DataFrame containing the DVH for the row
    """

    if not row.modality == "RTDOSE":
        raise ValueError("Row is not an RTDOSE")

    if isinstance(struct_hash, str):
        struct_hash = [struct_hash]

    row_path = Path(row.path)

    config = PyDicerConfig()

    # If the working directory is configured and the row_path isn't relative to it, join it.
    if config is not None:
        try:
            row_path.relative_to(config.get_working_dir())
        except ValueError:
            row_path = config.get_working_dir().joinpath(row_path)

    df_result = pd.DataFrame(columns=["patient", "struct_hash", "label", "cc", "mean"])
    for dvh_file in row_path.glob("dvh_*.csv"):
        if struct_hash is not None:
            file_struct_hash = dvh_file.name.replace("dvh_", "").replace(".csv", "")

            if file_struct_hash not in struct_hash:
                continue

        col_types = {"patient": str, "struct_hash": str, "label": str, "dose_hash": str}
        df_dvh = pd.read_csv(dvh_file, index_col=0, dtype=col_types)
        df_result = pd.concat([df_result, df_dvh])

    df_result.sort_values(["patient", "struct_hash", "label"], inplace=True)
    df_result.reset_index(drop=True, inplace=True)

    # Change the type of the columns which indicate the dose bins, useful for dose metric
    # computation later
    df_result.columns = [float(c) if "." in c else c for c in df_result.columns]

    return df_result


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

        col_types = {"patient_id": str, "hashed_uid": str}
        df_converted = pd.read_csv(converted_csv, index_col=0, dtype=col_types)
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
    """Reads the SimpleITK Image object given a converted dataframe row.

    Args:
        row (pd.Series): The row of the data frame for which to load the SimpleITK Image.

    Returns:
        SimpleITK.Image: The loaded image. Returns None if the image was not found.
    """

    object_path = Path(row.path)

    nifti_path = object_path.joinpath(f"{row.modality}.nii.gz")

    if not nifti_path.exists():
        logger.warning("Unable to load Nifti at %s", nifti_path)
        return None

    return sitk.ReadImage(str(nifti_path))


def get_iterator(iterable, length=None, unit="it", name=None):
    """Get the appropriate iterator based on the level of verbosity configured.

    Args:
        iterable (iterable): The list or iterable to iterate over.
        length (int, optional): The length of the iterator. If None, the len() functio will be used
          to determine the length (only works for list/tuple). Defaults to None.
        unit (str, optional): The unit string to display in the progress bar. Defaults to "it".
        name (str, optional): The name to display in the progress bar. Defaults to None.

    Returns:
        iterable: The appropriate iterable object.
    """

    config = PyDicerConfig()

    iterator = iterable
    if config.get_config("verbosity") == 0:
        if length is None:
            length = len(iterable)

        iterator = tqdm.tqdm(
            iterable,
            total=length,
            unit=unit,
            postfix=name,
        )

    return iterator


def map_structure_name(struct_name, struct_map_dict):
    """Function to map a structure's name according to a mapping dictionary

    Args:
        struct_name (str): the structure name to be mapped. If the name is remapped according to the
        mapping file, then the structure NifTi file is renamed with the mapped name
        struct_map_dict (dict): the mapping dictionary

    Returns:

        str: the mapped structure name
    """
    # Check if the structure name needs to be mapped
    mapped_struct_name_set = {i for i in struct_map_dict if struct_name in struct_map_dict[i]}

    # If not true, then either the structure name is already in mapped form, or the structure name
    # is not being captured in the specific mapping dictionary
    if len(mapped_struct_name_set) > 0:
        return mapped_struct_name_set.pop()

    return struct_name


def get_structures_linked_to_dose(working_directory: Path, dose_row: pd.Series) -> pd.DataFrame:
    """Get the structure sets which are linked to a dose object.

    Args:
        working_directory (Path): The PyDicer working folder.
        dose_row (pd.Series): The row from the converted data describing the dose object.

    Returns:
        pd.DataFrame: The data frame containing structure sets linked to row.
    """
    # Currently doses are linked via: plan -> struct -> image
    df_converted = read_converted_data(working_directory)

    # Find the linked plan
    df_linked_plan = df_converted[
        df_converted["sop_instance_uid"] == dose_row.referenced_sop_instance_uid
    ]

    if len(df_linked_plan) == 0:
        logger.warning("No linked plans found for dose: %s", dose_row.sop_instance_uid)

    # Find the linked structure set
    df_linked_struct = None
    if len(df_linked_plan) > 0:
        plan_row = df_linked_plan.iloc[0]
        df_linked_struct = df_converted[
            df_converted["sop_instance_uid"] == plan_row.referenced_sop_instance_uid
        ]

    # Also link via Frame of Reference
    df_for_linked = df_converted[
        (df_converted["modality"] == "RTSTRUCT") & (df_converted["for_uid"] == dose_row.for_uid)
    ]

    if df_linked_struct is None:
        df_linked_struct = df_for_linked
    else:
        df_linked_struct = pd.concat([df_linked_struct, df_for_linked])

    # Drop in case a structure was linked twice
    df_linked_struct = df_linked_struct.drop_duplicates()

    return df_linked_struct


def add_structure_name_mapping(
    mapping_dict: dict,
    mapping_id: str = DEFAULT_MAPPING_ID,
    working_directory: Path = None,
    patient_id: str = None,
    structure_set_row: pd.Series = None,
):
    """Specify a structure name mapping dictionary object where keys are the standardised structure
    names and value is a list of strings of various structure names to map to the standard name.

    If a `structure_set_row` is provided, the mapping will be stored only for that specific
    structure. Otherwise, `working_directory` must be provided, then it will be stored at project
    level by default, or at the patient level if `patient_id` is also provided.

    Args:
        mapping_dict (dict): Dictionary object with the standardised structure name (str) as the
          key and a list of the various structure names to map as the value.
        mapping_id (str, optional): The ID to refer to this mapping as. Defaults to
          DEFAULT_MAPPING_ID.
        working_directory (Path, optional): The working directory for this project Required if
          `structure_set_row` is None. Defaults to None.
        patient_id (str, optional): The ID of the patient to which this mapping belongs.
          Defaults to None.
        structure_set_row (pd.Series, optional): The row of the converted structure set to which
          this mapping belongs. Defaults to None.

    Raises:
        SystemError: Ensure working_directory or structure_set is provided.
        ValueError: All keys in mapping dictionary must be of type `str`.
        ValueError: All values in mapping dictionary must be a list of `str` entries.
    """

    mapping_path_base = None
    mapping_level = "project"

    if structure_set_row is not None:
        # Mapping for specific structure set
        logger.info(
            "Adding mapping %s for structure set %s", mapping_id, structure_set_row.hashed_uid
        )

        mapping_path_base = Path(structure_set_row.path)
        mapping_level = "stucture_set"

    elif working_directory is not None:
        # Mapping at a higher level, set the appropriate directory based on parameters passed in

        if patient_id is None:
            # Project wide, store this in the hiddel .pydicer directory
            mapping_path_base = working_directory.joinpath(".pydicer")
            mapping_level = "project"
        else:
            # Patient specific, store this in the patient directory
            mapping_path_base = working_directory.joinpath(
                CONVERTED_DIR_NAME, patient_id, "structures"
            )
            mapping_level = "patient"

    else:
        raise SystemError("working_directory or structure_set_row must be provided")

    # Perform a few checks on the mapping dict
    for k in mapping_dict.keys():
        if not isinstance(k, str):
            raise ValueError("All keys in mapping dictionary must be of type str")

        values_valid = isinstance(mapping_dict[k], list)
        for value in mapping_dict[k]:
            if not isinstance(value, str):
                values_valid = False
                break

        if not values_valid:
            raise ValueError(
                "All values in mapping dictionary must be a list of str entries "
                "(e.g. {'Lung_L': ['Left_Lung', 'LeftLung', 'lung_l']})"
            )

    logger.info("Adding mapping for %s in %s", mapping_level, mapping_path_base)

    mapping_path = mapping_path_base.joinpath(".structure_set_mappings")
    mapping_path.mkdir(exist_ok=True)

    # Create the mapping file
    with open(
        mapping_path.joinpath(f"{mapping_id}.json"), "w", encoding="utf8"
    ) as structures_map_file:
        json.dump(mapping_dict, structures_map_file, ensure_ascii=False, indent=4)


def download_and_extract_zip_file(zip_url, output_directory):
    """Downloads a zip file from the URL specified and extracts the contents to the output
    directory.

    Args:
        zip_url (str): The URL of the zip file.
        output_directory (str|pathlib.Path): The path in which to extract the contents.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        temp_file = temp_dir.joinpath("tmp.zip")

        with urllib.request.urlopen(zip_url) as dl_file:
            with open(temp_file, "wb") as out_file:
                out_file.write(dl_file.read())

        with zipfile.ZipFile(temp_file, "r") as zip_ref:
            zip_ref.extractall(output_directory)


def fetch_converted_test_data(working_directory=None, dataset="HNSCC"):
    """Fetch some public data which has already been converted using PyDicer.
    Useful for unit testing as well as examples.

    Args:
        working_directory (str|pathlib.Path, optional): The working directory in which to
          place the test data. Defaults to None.
        dataset (str, optional): The name of the dataset to fetch, either HNSCC or LCTSC.
          Defaults to "HNSCC".

    Returns:
        pathlib.Path: The path to the working directory.
    """

    if working_directory is None:
        working_directory = Path(".")
        working_directory.joinpath(dataset)

    working_directory = Path(working_directory)

    if working_directory.exists():
        logger.warning("Working directory %s aready exists, won't download test data.")
        return working_directory

    if dataset == "HNSCC":
        zip_url = "https://zenodo.org/record/8237552/files/HNSCC_pydicer.zip"
        working_name = "testdata"
    elif dataset == "LCTSC":
        zip_url = "https://zenodo.org/records/10005835/files/LCTSC_pydicer.zip"
        working_name = "LCTSC"
    else:
        raise ValueError(f"Unknown dataset {dataset}")

    with tempfile.TemporaryDirectory() as temp_dir:
        output_directory = Path(temp_dir).joinpath("output")
        download_and_extract_zip_file(zip_url, output_directory)
        shutil.copytree(output_directory.joinpath(working_name), working_directory)

    return working_directory


def copy_doc(copy_func, remove_args=None):
    """Copies the doc string of the given function to another.
    This function is intended to be used as a decorator.

    Remove args listed in `remove_args` from the docstring.

    This function was adapted from:
    https://stackoverflow.com/questions/68901049/copying-the-docstring-of-function-onto-another-function-by-name

    .. code-block:: python3

        def foo():
            '''This is a foo doc string'''
            ...

        @copy_doc(foo)
        def bar():
            ...

    """

    if remove_args is None:
        remove_args = []

    def wrapped(func):
        func.__doc__ = copy_func.__doc__

        for arg in remove_args:
            func.__doc__ = "\n".join(
                [line for line in func.__doc__.split("\n") if not line.strip().startswith(arg)]
            )

        return func

    return wrapped
