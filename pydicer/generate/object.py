import logging

import pandas as pd
import SimpleITK as sitk
from pydicer.constants import CONVERTED_DIR_NAME

from pydicer.utils import read_converted_data

logger = logging.getLogger(__name__)


def add_object(
    working_directory,
    object_id,
    patient_id,
    object_type,
    modality,
    for_uid=None,
    referenced_sop_instance_uid=None,
    datasets=None,
):
    """Add a generated object to the project.

    Args:
        working_directory (pathlib.Path): The working directory of the project.
        object_id (str): The unique ID of the new object.
        patient_id (str): The ID of the patient for which the object is being added.
        object_type (str): The type of object, must be one of "image", "structure", "plan" or
            "dose".
        modality (str): The modality of the object (as per the DICOM standard)
        for_uid (str, optional): The Frame of Reference UID. Defaults to None.
        referenced_sop_instance_uid (str, optional): The SOP Instance UID of the object referenced
            by the generated object. Defaults to None.
        datasets (list|str, optional): The name(s) of the dataset(s) to add the object to. Defaults
            to None.

    Raises:
        ValueError: Raised in object_type is not "image", "structure", "plan" or "dose".
        ValueError: Raised if the patient does not exist in the project.
        SystemError: Raised if the generated object does not yet exist on the file system.
        SystemError: Raised if an object with this ID has already exists in the project.
    """

    # Make sure the object type is one we expect
    if object_type not in ["image", "structure", "plan", "dose"]:
        raise ValueError("Object type must be one of: 'image', 'structure', 'plan' or 'dose'")

    # Make sure the patient directory exists
    patient_directory = working_directory.joinpath(CONVERTED_DIR_NAME, patient_id)
    if not patient_directory.exists():
        raise ValueError(
            f"Patient with ID {patient_id} does not exist. Objects can only be added "
            "for existing patients."
        )

    # The data object should already have been added to the file system, raise error if not.
    data_object_path = patient_directory.joinpath(f"{object_type}s", object_id)
    if not data_object_path.exists():
        raise SystemError(
            "Generated object does not yet exist on the file system. Create a "
            f"folder at: {data_object_path} containing the appropriate files. The re-run this "
            "function."
        )

    # Check that an object with this ID hasn't already been added to the dataset
    df_converted = read_converted_data(working_directory, patients=[patient_id])
    if len(df_converted[df_converted.hashed_uid == object_id]) > 0:
        raise SystemError(f"An object with ID {object_id} already exists for this patient!")

    # Do a few checks about the files within the data object directory. Won't raise any errors here
    # but will log warnings if something doesn't seem right
    if object_type == "image":
        if not data_object_path.joinpath("f{modality}.nii.gz"):
            logger.warning(
                "Expecting dose to be stored as a file named "
                "%s.nii.gz within the %s directory.",
                modality,
                data_object_path,
            )

    if object_type == "structure":
        if list(len(data_object_path.glob("*.nii.gz"))) == 0:
            logger.warning(
                "Expecting structures to be stored as a files named "
                "[struct_name].nii.gz within the %s directory.",
                data_object_path,
            )

    if object_type == "dose":
        if not data_object_path.joinpath("f{modality}.nii.gz"):
            logger.warning(
                "Expecting dose to be stored as a file named "
                "%s.nii.gz within the %s directory.",
                modality,
                data_object_path,
            )

    # Everything seems OK, so we'll add the data object to the set of converted data objects for
    # this patient
    entry = {
        "sop_instance_uid": object_id,
        "hashed_uid": object_id,
        "modality": modality,
        "patient_id": patient_id,
        "series_uid": object_id,
        "for_uid": for_uid if for_uid is not None else "",
        "referenced_sop_instance_uid": referenced_sop_instance_uid
        if referenced_sop_instance_uid is not None
        else "",
        "path": str(data_object_path.relative_to(working_directory)),
    }

    df_converted = pd.concat([df_converted, pd.DataFrame([entry])])
    df_converted.to_csv(patient_directory.joinpath("converted.csv"))

    # Now loop through each dataset and add in there. If the dataset doesn't exist or the patient
    # isn't in the dataset it won't be added and the user will be warned.
    if datasets is None:
        datasets = []

    if isinstance(datasets, str):
        datasets = [datasets]

    for dataset in datasets:

        # Make sure the dataset directory exists
        dataset_directory = working_directory.joinpath(dataset)
        if not dataset_directory.exists():
            logger.warning("Dataset with name %s doesn't exist", dataset)
            continue

        # Make sure the patient directory exists
        patient_directory = dataset_directory.joinpath(patient_id)
        if not patient_directory.exists():
            logger.warning("Patient %s doesn't exist in dataset %s", patient_id, dataset)
            continue

        df_converted = read_converted_data(
            working_directory, dataset_name=dataset, patients=[patient_id]
        )

        df_converted = pd.concat([df_converted, pd.DataFrame([entry])])
        df_converted.to_csv(patient_directory.joinpath("converted.csv"))


# Add image
# def add_image_object(
#     working_directory, image, image_id, patient_id, modality, for_uid=None, datasets=None
# ):

# create the folder

# save the image

# add the object to pydicer


# Add structures
# def add_structure_object(
#     working_directory, structures, structure_id, patient_id, linked_image=None, datasets=None
# ):

# create the folder

# save off the structures

# add the object to pydicer


def add_dose_object(working_directory, dose, dose_id, patient_id, linked_plan=None, datasets=None):
    """Add a generated dose object to the project.

    Args:
        working_directory (pathlib.Path): The project working directory.
        dose (sitk.Image): A SimpleITK.Image of the dose volume to add.
        dose_id (str): The unique ID of the new dose object.
        patient_id (str): The ID of the patient of this object.
        linked_plan (str|pd.Series, optional): The hashed_uidor the Pandas DataFrame row of the
            RTPLAN object to link to. If None the new object won't be linked. Defaults to None.
        datasets (list|str, optional): The name(s) of the dataset(s) to add the object to. Defaults
            to None.

    Raises:
        ValueError: Raised then the patient ID does not exist
        SystemError: Raised when a linked_plan is provided but can't be found for this patient.
    """

    # Check that the patient directory exists
    patient_directory = working_directory.joinpath(CONVERTED_DIR_NAME, patient_id)
    if not patient_directory.exists():
        raise ValueError(f"Patient with ID {patient_id} does not exist.")

    object_type = "dose"

    # If we have linked data, use the for_uid and reference that sop_instance_uid
    for_uid = None
    ref_sop_uid = None
    if linked_plan is not None:

        if isinstance(linked_plan, str):
            # Find the entry
            df_converted = read_converted_data(working_directory, patients=[patient_id])

            df_linked = df_converted[df_converted.hashed_uid == linked_plan]

            if not len(df_linked) == 1:
                raise SystemError(f"Expecting one linked_plan object but found {len(df_linked) }")

            linked_plan = df_linked.iloc[0]

        for_uid = linked_plan.for_uid
        ref_sop_uid = linked_plan.sop_instance_uid

    dose_object_path = patient_directory.joinpath(f"{object_type}s", dose_id)
    dose_object_path.mkdir(
        exist_ok=True,
    )

    # Save the data object in the dose_object_path
    sitk.WriteImage(dose, str(dose_object_path.joinpath("RTDOSE.nii.gz")))

    # Add the object to pydicer
    add_object(
        working_directory,
        dose_id,
        patient_id,
        object_type,
        "RTDOSE",
        for_uid=for_uid,
        referenced_sop_instance_uid=ref_sop_uid,
        datasets=datasets,
    )
