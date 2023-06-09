import logging
import tempfile
import copy
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import SimpleITK as sitk
import pydicom
from matplotlib import cm

from platipy.dicom.io.rtdose_to_nifti import convert_rtdose
from pydicer.config import PyDicerConfig

from pydicer.convert.pt import convert_dicom_to_nifti_pt
from pydicer.convert.rtstruct import convert_rtstruct, write_nrrd_from_mask_directory
from pydicer.convert.headers import convert_dicom_headers
from pydicer.utils import hash_uid, read_preprocessed_data, get_iterator
from pydicer.quarantine.treat import copy_file_to_quarantine

from pydicer.constants import (
    CONVERTED_DIR_NAME,
    PYDICER_DIR_NAME,
    RT_DOSE_STORAGE_UID,
    RT_PLAN_STORAGE_UID,
    RT_STRUCTURE_STORAGE_UID,
    CT_IMAGE_STORAGE_UID,
    PET_IMAGE_STORAGE_UID,
)
from pydicer.logger import PatientLogger

logger = logging.getLogger(__name__)

OBJECT_TYPES = {
    "images": [CT_IMAGE_STORAGE_UID, PET_IMAGE_STORAGE_UID],
    "structures": [RT_STRUCTURE_STORAGE_UID],
    "plans": [RT_PLAN_STORAGE_UID],
    "doses": [RT_DOSE_STORAGE_UID],
}

DATA_OBJECT_COLUMNS = [
    "sop_instance_uid",
    "hashed_uid",
    "modality",
    "patient_id",
    "series_uid",
    "for_uid",
    "referenced_sop_instance_uid",
    "path",
]


def get_object_type(sop_class_uid):
    """Get the type of the object (used for the output path)

    Args:
        sop_class_uid (str): The SOP Class UID of the object

    Returns:
        str: The object type
    """

    object_type = "other"
    for ot, sops in OBJECT_TYPES.items():
        if sop_class_uid in sops:
            object_type = ot

    return object_type


def handle_missing_slice(files):
    """Function to interpolate missing slices in an image

    Example usage:

    .. code-block:: python

        from pydicer.convert.data import handle_missing_slice

        input_dic = [
            {
                "file_path" : "path/to/dicom_file_1.dcm",
                "slice_location: -100
            },
            {
                "file_path" : "path/to/dicom_file_2.dcm",
                "slice_location: -98
            },
            {
                "file_path" : "path/to/dicom_file_3.dcm",
                "slice_location: -96
            },
            ...
        ]
        file_paths_list = handle_missing_slices(input_dict)

    Args:
        df_files (pd.DataFrame|list): the DataFrame which was produced by PreprocessData
        or list of filepaths to dicom slices

    Returns:
        file_paths(list): a list of the interpolated file paths
    """

    if isinstance(files, pd.DataFrame):
        df_files = files
    elif isinstance(files, list):
        df_files = pd.DataFrame(files)

        df_files = df_files.sort_values(["slice_location"])
    else:
        raise ValueError("This function requires a Dataframe or list")

    # If duplicated slices locations are present in series, check if the first duplicates have the
    # same pixel data. If they do, then assume all are the same and drop the duplicates. Otherwise
    # we raise an error.
    df_duplicated = df_files[df_files["slice_location"].duplicated()]
    if len(df_duplicated) > 0:
        check_slice_location = df_duplicated.iloc[0].slice_location
        df_check_duplicates = df_files[df_files["slice_location"] == check_slice_location]

        pix_array = None
        for _, row in df_check_duplicates.iterrows():
            this_pix_array = pydicom.read_file(row.file_path, force=True).pixel_array

            if pix_array is None:
                pix_array = this_pix_array
            else:
                if not np.allclose(pix_array, this_pix_array):
                    raise (
                        ValueError(
                            f"{len(df_check_duplicates)} slices at location "
                            f"{check_slice_location} containing different pixel data."
                        )
                    )

        logger.warning("Duplicate slices detected, pixel array the same so dropping duplicates")
        df_files = df_files.drop_duplicates(subset=["slice_location"])

    temp_dir = Path(tempfile.mkdtemp())

    slice_location_diffs = np.diff(df_files.slice_location.to_numpy(dtype=float))

    unique_slice_diffs, _ = np.unique(slice_location_diffs, return_counts=True)
    expected_slice_diff = unique_slice_diffs[0]

    # check to see if any slice thickness exceed 2% tolerance
    # this is conservative as missing slices would produce 100% differences
    slice_thickness_variations = ~np.isclose(slice_location_diffs, expected_slice_diff, rtol=0.02)

    if np.any(slice_thickness_variations):

        logger.warning("Missing DICOM slices found")

        # find where the missing slices are
        missing_indices = np.where(slice_thickness_variations)[0]

        for missing_index in missing_indices:

            num_missing_slices = int(slice_location_diffs[missing_index] / expected_slice_diff) - 1

            # locate nearest DICOM files to the missing slices
            prior_dcm_file = df_files.iloc[missing_index]["file_path"]
            next_dcm_file = df_files.iloc[missing_index + 1]["file_path"]

            prior_dcm = pydicom.read_file(prior_dcm_file)
            next_dcm = pydicom.read_file(next_dcm_file)

            working_dcm = copy.deepcopy(prior_dcm)

            prior_array = prior_dcm.pixel_array.astype(float)
            next_array = next_dcm.pixel_array.astype(float)

            for missing_slice in range(num_missing_slices):

                # TODO add other interp options (cubic)
                interp_array = np.array(
                    prior_array
                    + ((missing_slice + 1) / (num_missing_slices + 1))
                    * (next_array - prior_array),
                    next_array.dtype,
                )

                # write a copy to a temporary DICOM file
                working_dcm.PixelData = interp_array.astype(prior_dcm.pixel_array.dtype).tobytes()

                # compute spatial information
                image_orientation = np.array(prior_dcm.ImageOrientationPatient, dtype=float)

                image_plane_normal = np.cross(image_orientation[:3], image_orientation[3:])

                image_position_patient = np.array(
                    np.array(prior_dcm.ImagePositionPatient)
                    + (
                        ((missing_slice + 1) / (num_missing_slices + 1))
                        * (
                            np.array(next_dcm.ImagePositionPatient)
                            - np.array(prior_dcm.ImagePositionPatient)
                        )
                    )
                )

                slice_location = (image_position_patient * image_plane_normal)[2]

                # insert new spatial information into interpolated slice
                working_dcm.SliceLocation = slice_location
                working_dcm.ImagePositionPatient = image_position_patient.tolist()

                # write interpolated slice to DICOM
                interp_dcm_file = temp_dir / f"{slice_location}.dcm"
                pydicom.write_file(interp_dcm_file, working_dcm)

                # insert into dataframe
                interp_df_row = dict(df_files.iloc[missing_index])
                interp_df_row["slice_location"] = slice_location
                interp_df_row["file_path"] = str(interp_dcm_file)

                df_files = pd.concat([df_files, pd.DataFrame([interp_df_row])])
                df_files.sort_values(by="slice_location", inplace=True)

    return df_files.file_path.tolist()


def link_via_frame_of_reference(for_uid, df_preprocess):
    """Find the image series linked to this FOR

    Args:
        for_uid (str): The Frame of Reference UID
        df_preprocess (pd.DataFrame): The DataFrame containing the preprocessed DICOM data.

    Returns:
        pd.DataFrame: DataFrame of the linked series entries
    """

    df_linked_series = df_preprocess[df_preprocess.for_uid == for_uid]

    # Find the image series to link to in this order of perference
    modality_prefs = ["CT", "MR", "PT"]

    df_linked_series = df_linked_series[df_linked_series.modality.isin(modality_prefs)]
    df_linked_series.loc[:, "modality"] = df_linked_series.modality.astype("category")
    df_linked_series.modality.cat.set_categories(modality_prefs, inplace=True)
    df_linked_series.sort_values(["modality"])

    return df_linked_series


class ConvertData:
    """
    Class that facilitates the conversion of the data into its intended final type

    Args:
        - working_directory (str|pathlib.Path, optional): Main working directory for pydicer.
            Defaults to ".".
    """

    def __init__(self, working_directory="."):
        self.working_directory = Path(working_directory)
        self.pydicer_directory = working_directory.joinpath(PYDICER_DIR_NAME)
        self.output_directory = working_directory.joinpath(CONVERTED_DIR_NAME)

    def add_entry(self, entry):
        """Add an entry of a converted data object to the patient's converted dataframe.

        Args:
            entry (dict): A dictionary object describing the object converted.
        """

        patient_id = entry["patient_id"]
        hashed_uid = entry["hashed_uid"]

        # Load the converted dataframe for this patient if it exists, otherwise create a new one
        patient_directory = self.output_directory.joinpath(patient_id)
        converted_df_path = patient_directory.joinpath("converted.csv")
        if converted_df_path.exists():
            col_types = {"patient_id": str, "hashed_uid": str}
            df_pat_data = pd.read_csv(converted_df_path, index_col=0, dtype=col_types)
            df_pat_data = df_pat_data.reset_index(drop=True)
        else:
            df_pat_data = pd.DataFrame(columns=DATA_OBJECT_COLUMNS)

        # If this entry already existed, replace that row in the dataframe. Otherwise append.
        if len(df_pat_data[df_pat_data.hashed_uid == hashed_uid]) > 0:
            for c in entry:
                df_pat_data.loc[df_pat_data.hashed_uid == hashed_uid, c] = entry[c]
        else:
            df_pat_data = pd.concat([df_pat_data, pd.DataFrame([entry])])

        logger.info(
            "Successfully converted %s object with hashed UID: %s",
            entry["modality"],
            entry["hashed_uid"],
        )

        # Save the patient converted dataframe
        df_pat_data = df_pat_data.reset_index(drop=True)
        df_pat_data.to_csv(converted_df_path)

    def convert(self, patient=None, force=True):
        """Converts the DICOM which was preprocessed into the pydicer output directory.

        Args:
            patient (str|list, optional): Patient ID or list of patient IDs to convert. Defaults to
              None.
            force (bool, optional): When True objects will be converted even if the output files
              already exist. Defaults to True.
        """

        # Create the output directory if it hasn't already been created
        self.output_directory.mkdir(exist_ok=True)

        # Load the preprocessed data
        df_preprocess = read_preprocessed_data(self.working_directory)

        config = PyDicerConfig()

        if patient is not None:
            if not isinstance(patient, list):
                patient = [patient]

            df_preprocess = df_preprocess[df_preprocess["patient_id"].isin(patient)]

        for key, df_files in get_iterator(
            df_preprocess.groupby(["patient_id", "modality", "series_uid"]),
            unit="objects",
            name="convert",
        ):

            patient_id, _, series_uid = key

            logger.info("Converting data for patient: %s", patient_id)

            patient_directory = self.output_directory.joinpath(patient_id)

            patient_logger = PatientLogger(patient_id, self.output_directory, force=False)

            # Grab the sop_class_uid, modality and for_uid (should be the same for all files in
            # series)
            sop_class_uid = df_files.sop_class_uid.unique()[0]
            modality = df_files.modality.unique()[0]
            for_uid = df_files.for_uid.unique()[0]

            # Use the SOPInstanceUID as a hash for the output object. In case of a series (where
            # multiple DICOM objects make up the converted object) then use the SOPInstanceUID of
            # the first object.
            sop_instance_uid = df_files.sop_instance_uid.unique()[0]
            sop_instance_hash = hash_uid(sop_instance_uid)

            # Determine the output type to decide in which directory the object should be placed
            object_type = get_object_type(sop_class_uid)

            output_dir = patient_directory.joinpath(object_type, sop_instance_hash)

            entry = {
                "sop_instance_uid": sop_instance_uid,
                "hashed_uid": sop_instance_hash,
                "modality": modality,
                "patient_id": patient_id,
                "series_uid": series_uid,
                "for_uid": for_uid,
            }

            try:
                if sop_class_uid == CT_IMAGE_STORAGE_UID:

                    if not output_dir.exists() or force:

                        # Only convert if it doesn't already exist or if force is True

                        if config.get_config("interp_missing_slices"):
                            series_files = handle_missing_slice(df_files)
                        else:
                            # TODO Handle inconsistent slice spacing
                            error_log = """Slice Locations are not evenly spaced. Set
                                interp_missing_slices to True to interpolate slices."""
                            patient_logger.log_module_error(
                                "convert", sop_instance_hash, error_log
                            )

                        output_dir.mkdir(exist_ok=True, parents=True)

                        series_files = [str(f) for f in series_files]
                        series = sitk.ReadImage(series_files)

                        nifti_file = output_dir.joinpath("CT.nii.gz")
                        sitk.WriteImage(series, str(nifti_file))
                        logger.debug("Writing CT Image Series to: %s", nifti_file)

                        json_file = output_dir.joinpath("metadata.json")
                        convert_dicom_headers(
                            series_files[0],
                            str(nifti_file.relative_to(self.output_directory)),
                            json_file,
                        )

                    entry["path"] = str(output_dir.relative_to(self.working_directory))

                    self.add_entry(entry)
                    patient_logger.eval_module_process("convert", sop_instance_hash)

                elif sop_class_uid == RT_STRUCTURE_STORAGE_UID:

                    # If we have multiple structure sets with the same sop_instance_uid we'll just
                    # drop them
                    df_files = df_files.drop_duplicates(subset=["sop_instance_uid"])

                    rt_struct_file = df_files.iloc[0]

                    # Get the linked image
                    # Need to disable this pylint check here only, seems to be a bug in
                    # pylint/pandas
                    # pylint: disable=unsubscriptable-object
                    df_linked_series = df_preprocess[
                        df_preprocess["series_uid"] == rt_struct_file.referenced_uid
                    ]

                    # If not linked via referenced UID, then try to link via FOR
                    if len(df_linked_series) == 0:
                        for_uid = rt_struct_file.referenced_for_uid
                        df_linked_series = link_via_frame_of_reference(for_uid, df_preprocess)

                    # Check that the linked series is available
                    # TODO handle rendering the masks even if we don't have an image series it's
                    # linked to
                    if len(df_linked_series) == 0:
                        error_log = "Series Referenced by RTSTRUCT not found"
                        patient_logger.log_module_error("convert", sop_instance_hash, error_log)

                    if not output_dir.exists() or force:

                        # Only convert if it doesn't already exist or if force is True
                        output_dir.mkdir(exist_ok=True, parents=True)

                        img_row = df_linked_series.iloc[0]
                        hashed_linked_id = hash_uid(img_row.sop_instance_uid)
                        linked_nifti_file = self.output_directory.joinpath(
                            patient_id,
                            "images",
                            hashed_linked_id,
                            f"{img_row.modality}.nii.gz",
                        )
                        img_file = sitk.ReadImage(str(linked_nifti_file))

                        convert_rtstruct(
                            img_file,
                            rt_struct_file.file_path,
                            prefix="",
                            output_dir=output_dir,
                            output_img=None,
                            spacing=None,
                        )

                        if config.get_config("generate_nrrd"):
                            nrrd_file = output_dir.joinpath("STRUCTURE_SET.nrrd")
                            logger.info("Saving structures in nrrd format: %s", nrrd_file)
                            write_nrrd_from_mask_directory(
                                output_dir,
                                nrrd_file,
                                colormap=cm.get_cmap(config.get_config("nrrd_colormap")),
                            )

                        # Save JSON
                        json_file = output_dir.joinpath("metadata.json")
                        convert_dicom_headers(
                            rt_struct_file.file_path,
                            str(output_dir.relative_to(self.output_directory)),
                            json_file,
                        )

                    entry["path"] = str(output_dir.relative_to(self.working_directory))
                    entry["for_uid"] = rt_struct_file.referenced_for_uid

                    # Find the SOP Instance UID of the CT image to link to
                    entry[
                        "referenced_sop_instance_uid"
                    ] = df_linked_series.sop_instance_uid.unique()[0]

                    self.add_entry(entry)
                    patient_logger.eval_module_process(
                        "convert",
                        sop_instance_hash,
                    )

                elif sop_class_uid == PET_IMAGE_STORAGE_UID:

                    if not output_dir.exists() or force:

                        # Only convert if it doesn't already exist or if force is True

                        series_files = df_files.file_path.tolist()
                        series_files = [str(f) for f in series_files]

                        output_dir.mkdir(exist_ok=True, parents=True)
                        nifti_file = output_dir.joinpath("PT.nii.gz")

                        convert_dicom_to_nifti_pt(
                            series_files,
                            nifti_file,
                            default_patient_weight=config.get_config("default_patient_weight"),
                        )

                        json_file = output_dir.joinpath("metadata.json")
                        convert_dicom_headers(
                            series_files[0],
                            str(nifti_file.relative_to(self.output_directory)),
                            json_file,
                        )

                    entry["path"] = str(output_dir.relative_to(self.working_directory))

                    self.add_entry(entry)
                    patient_logger.eval_module_process("convert", sop_instance_hash)

                elif sop_class_uid == RT_PLAN_STORAGE_UID:

                    # If we have multiple plans with the same sop_instance_uid we'll just drop them
                    df_files = df_files.drop_duplicates(subset=["sop_instance_uid"])

                    # No Nifti to create here, just save the JSON

                    # If there are multiple RTPLANs in the same series then just save them all
                    for _, rt_plan_file in df_files.iterrows():

                        sop_instance_hash = hash_uid(rt_plan_file.sop_instance_uid)

                        # Update the output directory for this plan
                        output_dir = patient_directory.joinpath(object_type, sop_instance_hash)

                        if not output_dir.exists() or force:

                            # Only convert if it doesn't already exist or if force is True
                            output_dir.mkdir(exist_ok=True, parents=True)

                            json_file = output_dir.joinpath("metadata.json")

                            convert_dicom_headers(rt_plan_file.file_path, "", json_file)

                        entry["sop_instance_uid"] = rt_plan_file.sop_instance_uid
                        entry["hashed_uid"] = sop_instance_hash
                        entry["referenced_sop_instance_uid"] = rt_plan_file.referenced_uid
                        entry["path"] = str(output_dir.relative_to(self.working_directory))

                        self.add_entry(entry)
                        patient_logger.eval_module_process("convert", sop_instance_hash)

                elif sop_class_uid == RT_DOSE_STORAGE_UID:

                    # If we have multiple doses with the same sop_instance_uid we'll just drop them
                    df_files = df_files.drop_duplicates(subset=["sop_instance_uid"])

                    # If there are multiple RTDOSEs in the same series then just save them all
                    for _, rt_dose_file in df_files.iterrows():

                        sop_instance_hash = hash_uid(rt_dose_file.sop_instance_uid)

                        # Update the output directory for this plan
                        output_dir = patient_directory.joinpath(object_type, sop_instance_hash)

                        if not output_dir.exists() or force:

                            # Only convert if it doesn't already exist or if force is True
                            output_dir.mkdir(exist_ok=True, parents=True)

                            nifti_file = output_dir.joinpath("RTDOSE.nii.gz")
                            nifti_file.parent.mkdir(exist_ok=True, parents=True)
                            logger.debug("Writing RTDOSE to: %s", nifti_file)
                            convert_rtdose(
                                rt_dose_file.file_path, force=True, dose_output_path=nifti_file
                            )

                            json_file = output_dir.joinpath("metadata.json")
                            convert_dicom_headers(
                                rt_dose_file.file_path,
                                str(nifti_file.relative_to(self.output_directory)),
                                json_file,
                            )

                        entry["sop_instance_uid"] = rt_dose_file.sop_instance_uid
                        entry["hashed_uid"] = sop_instance_hash
                        entry["referenced_sop_instance_uid"] = rt_dose_file.referenced_uid
                        entry["path"] = str(output_dir.relative_to(self.working_directory))

                        self.add_entry(entry)

                        patient_logger.eval_module_process("convert", sop_instance_hash)
                else:
                    raise NotImplementedError(
                        "Unable to convert Series with SOP Class UID: {sop_class_uid} / "
                        f"Modality: {modality}"
                    )

            except Exception as e:  # pylint: disable=broad-except
                # Broad except ok here, since we will put these file into a
                # quarantine location for further inspection.
                logger.error(
                    "Unable to convert series for patient: %s with UID: %s", patient_id, series_uid
                )
                logger.exception(e)

                # Remove the output_dir if it had already been created
                if output_dir.exists():
                    shutil.rmtree(output_dir)

                for f in df_files.file_path.tolist():
                    logger.error(
                        "Error parsing file %s: %s. Placing file into Quarantine folder...", f, e
                    )
                    copy_file_to_quarantine(Path(f), self.working_directory, e)
                patient_logger.log_module_error("convert", sop_instance_hash, e)
