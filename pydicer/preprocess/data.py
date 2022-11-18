import logging
from pathlib import Path

import pandas as pd
import pydicom
import numpy as np

from pydicer.config import PyDicerConfig
from pydicer.constants import (
    DICOM_FILE_EXTENSIONS,
    PET_IMAGE_STORAGE_UID,
    PYDICER_DIR_NAME,
    RT_DOSE_STORAGE_UID,
    RT_PLAN_STORAGE_UID,
    RT_STRUCTURE_STORAGE_UID,
    CT_IMAGE_STORAGE_UID,
)
from pydicer.quarantine.treat import copy_file_to_quarantine
from pydicer.utils import read_preprocessed_data, get_iterator

logger = logging.getLogger(__name__)


class PreprocessData:
    """
    Class for preprocessing the data information into a dicionary that holds the data in a
    structured hierarchy

    Args:
        working_directory (Path): The pydicer working directory
    """

    def __init__(self, working_directory):
        self.working_directory = working_directory
        self.pydicer_directory = working_directory.joinpath(PYDICER_DIR_NAME)
        self.pydicer_directory.mkdir(exist_ok=True)

    def scan_file(self, file):
        """Scan a DICOM file.

        Args:
            file (pathlib.Path|str): The path to the file to scan.

        Returns:
            dict: Returns the dict object containing the scanned information. None if the file
              couldn't be scanned.
        """

        logger.debug("Scanning file %s", file)

        ds = pydicom.read_file(file, force=True)

        try:

            dicom_type_uid = ds.SOPClassUID

            res_dict = {
                "patient_id": ds.PatientID,
                "study_uid": ds.StudyInstanceUID,
                "series_uid": ds.SeriesInstanceUID,
                "modality": ds.Modality,
                "sop_class_uid": dicom_type_uid,
                "sop_instance_uid": ds.SOPInstanceUID,
                "file_path": str(file),
            }

            if "FrameOfReferenceUID" in ds:
                res_dict["for_uid"] = ds.FrameOfReferenceUID

            if dicom_type_uid == RT_STRUCTURE_STORAGE_UID:

                try:
                    referenced_series_uid = (
                        ds.ReferencedFrameOfReferenceSequence[0]
                        .RTReferencedStudySequence[0]
                        .RTReferencedSeriesSequence[0]
                        .SeriesInstanceUID
                    )
                    res_dict["referenced_uid"] = referenced_series_uid
                except AttributeError:
                    logger.warning("Unable to determine Reference Series UID")

                try:
                    # Check other tags for a linked DICOM
                    # e.g. ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
                    # Potentially, we should check each referenced
                    referenced_frame_of_reference_uid = ds.ReferencedFrameOfReferenceSequence[
                        0
                    ].FrameOfReferenceUID
                    res_dict["referenced_for_uid"] = referenced_frame_of_reference_uid
                except AttributeError:
                    logger.warning("Unable to determine Referenced Frame of Reference UID")

            elif dicom_type_uid == RT_PLAN_STORAGE_UID:

                try:
                    referenced_sop_instance_uid = ds.ReferencedStructureSetSequence[
                        0
                    ].ReferencedSOPInstanceUID
                    res_dict["referenced_uid"] = referenced_sop_instance_uid
                except AttributeError:
                    logger.warning("Unable to determine Reference Series UID")

            elif dicom_type_uid == RT_DOSE_STORAGE_UID:

                try:
                    referenced_sop_instance_uid = ds.ReferencedRTPlanSequence[
                        0
                    ].ReferencedSOPInstanceUID
                    res_dict["referenced_uid"] = referenced_sop_instance_uid
                except AttributeError:
                    logger.warning("Unable to determine Reference Series UID")

            elif dicom_type_uid in (CT_IMAGE_STORAGE_UID, PET_IMAGE_STORAGE_UID):

                image_position = np.array(ds.ImagePositionPatient, dtype=float)
                image_orientation = np.array(ds.ImageOrientationPatient, dtype=float)

                image_plane_normal = np.cross(image_orientation[:3], image_orientation[3:])

                slice_location = (image_position * image_plane_normal)[2]

                res_dict["slice_location"] = slice_location

            else:
                raise ValueError(f"Could not determine DICOM type {ds.Modality} {dicom_type_uid}.")

            logger.debug(
                "Successfully scanned DICOM file with SOP Instance UID: %s",
                res_dict["sop_instance_uid"],
            )

            return res_dict

        except Exception as e:  # pylint: disable=broad-except
            # Broad except ok here, since we will put these file into a
            # quarantine location for further inspection.
            logger.error("Unable to preprocess file: %s", file)
            logger.exception(e)
            copy_file_to_quarantine(file, self.working_directory, e)

        return None

    def preprocess(self, input_directory, force=True):
        """
        Function to preprocess information regarding the data located in an Input working directory

        Args:
            input_directory (Path|list): The directory (or list of directories) containing the
              DICOM input data
            force (bool, optional): When True, all files will be preprocessed. Otherwise only files
              not already scanned previously will be preprocessed. Defaults to True.

        Returns: res_dict (pd.DataFrame): containing a row for each DICOM file that was
           preprocessed, with the following columns:
            - patient_id: PatientID field from the DICOM header
            - study_uid: StudyInstanceUID field from the DICOM header
            - series_uid: SeriesInstanceUID field from the DICOM header
            - modality: Modailty field from the DICOM header
            - sop_class_uid: SOPClassUID field from the DICOM header
            - sop_instance_uid: SOPInstanceUID field from the DICOM header
            - for_uid: FrameOfReferenceUID field from the DICOM header
            - file_path: The path to the file (as a pathlib.Path object)
            - slice_location: The real-world location of the slice (used for imaging modalities)
            - referenced_uid: The SeriesUID referenced by this DICOM file for RTSTRUCT
              and RTDOSE, the SOPInstanceUID of the structure set referenced by an RTPLAN.
            - referenced_for_uid: The ReferencedFrameOfReferenceUID referenced by this DICOM file

        """

        if isinstance(input_directory, Path):
            input_directory = [input_directory]

        if not isinstance(input_directory, list):
            raise ValueError("input_directory must be of type pathlib.Path or list")

        preprocessed_csv_path = self.pydicer_directory.joinpath("preprocessed.csv")

        df = pd.DataFrame(
            columns=[
                "patient_id",
                "study_uid",
                "series_uid",
                "modality",
                "sop_class_uid",
                "sop_instance_uid",
                "for_uid",
                "file_path",
                "slice_location",
                "referenced_uid",
                "referenced_for_uid",
            ]
        )

        files = []

        config = PyDicerConfig()
        for directory in input_directory:
            if config.get_config("enforce_dcm_ext"):
                for ext in DICOM_FILE_EXTENSIONS:
                    files += list(directory.glob(f"**/*.{ext}"))
            else:
                files += list(f for f in directory.glob("**/*") if not f.is_dir())

        # If we don't want to force preprocess and preprocesses files already exists, filter these
        # out
        if not force and preprocessed_csv_path.exists():

            logger.info("Not forcing preprocessing, will only scan unindexed files")

            df = read_preprocessed_data(self.working_directory)
            files_already_scanned = df.file_path.tolist()

            files = [f for f in files if str(f) not in files_already_scanned]

        logger.info("Found %d files to scan", len(files))

        result_list = []

        for f in get_iterator(files, unit="files", name="preprocess"):
            result = self.scan_file(f)
            if result is not None:
                result_list.append(result)

        df = pd.concat([df, pd.DataFrame(result_list)])

        # Sort the the DataFrame by the patient then series uid and the slice location, ensuring
        # that the slices are ordered correctly
        df = df.sort_values(["patient_id", "modality", "series_uid", "slice_location"])

        logger.info("Total of %d preprocessed DICOM files", len(df))

        # Save the Preprocessed DataFrame
        df.to_csv(preprocessed_csv_path)

        return df
