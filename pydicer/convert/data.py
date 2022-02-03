import logging
import tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pydicom

from platipy.dicom.io.rtdose_to_nifti import convert_rtdose

from pydicer.convert.pt import convert_dicom_to_nifti_pt
from pydicer.convert.rtstruct import convert_rtstruct, write_nrrd_from_mask_directory
from pydicer.convert.headers import convert_dicom_headers
from pydicer.utils import hash_uid
from pydicer.quarantine.treat import copy_file_to_quarantine

from pydicer.constants import (
    RT_DOSE_STORAGE_UID,
    RT_PLAN_STORAGE_UID,
    RT_STRUCTURE_STORAGE_UID,
    CT_IMAGE_STORAGE_UID,
    PET_IMAGE_STORAGE_UID,
)

logger = logging.getLogger(__name__)

# TODO make this user-selected
INTERPOLATE_MISSING_DATA = True


class ConvertData:
    """
    Class that facilitates the conversion of the data into its intended final type

    Args:
        - df_preprocess (pd.DataFrame): the DataFrame which was produced by PreprocessData
        - output_directory (str|pathlib.Path, optional): Directory in which to store converted data.
            Defaults to ".".
    """

    def __init__(self, df_preprocess, output_directory="."):
        self.df_preprocess = df_preprocess
        self.output_directory = Path(output_directory)

    def link_via_frame_of_reference(self, for_uid):
        """Find the image series linked to this FOR

        Args:
            for_uid (str): The Frame of Reference UID

        Returns:
            pd.DataFrame: DataFrame of the linked series entries
        """

        df_linked_series = self.df_preprocess[self.df_preprocess.for_uid == for_uid]

        # Find the image series to link to in this order of perference
        modality_prefs = ["CT", "MR", "PT"]

        df_linked_series = df_linked_series[df_linked_series.modality.isin(modality_prefs)]
        df_linked_series.loc[:, "modality"] = df_linked_series.modality.astype("category")
        df_linked_series.modality.cat.set_categories(modality_prefs, inplace=True)
        df_linked_series.sort_values(["modality"])

        return df_linked_series

    def convert(self, patient=None):
        """
        Function to convert the data into its intended form (eg. images into Nifti)
        """

        if patient is not None and not hasattr(patient, "__iter__"):
            patient = [patient]

        for key, df_files in self.df_preprocess.groupby(["patient_id", "series_uid"]):

            patient_id, series_uid = key

            if patient is not None and patient_id not in patient:
                continue

            # Grab the patied_id, study_uid, sop_class_uid and modality (should be the same for all
            # files in series)
            patient_id = df_files.patient_id.unique()[0]
            # study_uid = df_files.study_uid.unique()[0]
            sop_class_uid = df_files.sop_class_uid.unique()[0]
            modality = df_files.sop_class_uid.unique()[0]

            # Prepare some hashes of these UIDs to use for directory/file output paths
            # study_id_hash = hash_uid(study_uid)
            series_uid_hash = hash_uid(series_uid)

            try:
                # TODO abstract this interpolation, apply to other image modalities
                if sop_class_uid == CT_IMAGE_STORAGE_UID:
                    # Check that the slice location spacing is consistent, if not raise and error
                    # for now
                    slice_location_diffs = np.diff(df_files.slice_location.to_numpy(dtype=float))

                    unique_slice_diffs, _ = np.unique(slice_location_diffs, return_counts=True)
                    expected_slice_diff = unique_slice_diffs[0]

                    # check to see if any slice thickness exceed 2% tolerance
                    # this is conservative as missing slices would produce 100% differences
                    slice_thickness_variations = ~np.isclose(
                        slice_location_diffs, expected_slice_diff, rtol=0.02
                    )

                    if np.any(slice_thickness_variations):

                        logger.warning(
                            "Missing DICOM slices found: %s", df_files.iloc[0]["series_uid"]
                        )

                        temp_dir = Path(tempfile.mkdtemp())

                        # TODO Handle inconsistent slice spacing
                        if INTERPOLATE_MISSING_DATA:
                            # find where the missing slices are
                            missing_indices = np.where(slice_thickness_variations)[0]

                            for missing_index in missing_indices:

                                # locate nearest DICOM files to the missing slices
                                prior_dcm_file = df_files.iloc[missing_index]["file_path"]
                                next_dcm_file = df_files.iloc[missing_index + 1]["file_path"]

                                prior_dcm = pydicom.read_file(prior_dcm_file)
                                next_dcm = pydicom.read_file(next_dcm_file)

                                # TODO add other interp options (cubic)
                                interp_array = np.array(
                                    (prior_dcm.pixel_array + next_dcm.pixel_array) / 2,
                                    prior_dcm.pixel_array.dtype,
                                )

                                # write a copy to a temporary DICOM file
                                prior_dcm.PixelData = interp_array.tobytes()

                                # compute spatial information
                                image_orientation = np.array(
                                    prior_dcm.ImageOrientationPatient, dtype=float
                                )

                                image_plane_normal = np.cross(
                                    image_orientation[:3], image_orientation[3:]
                                )

                                image_position_patient = np.array(
                                    (
                                        np.array(prior_dcm.ImagePositionPatient)
                                        + np.array(next_dcm.ImagePositionPatient)
                                    )
                                    / 2,
                                )

                                slice_location = (image_position_patient * image_plane_normal)[2]

                                # insert new spatial information into interpolated slice
                                prior_dcm.SliceLocation = slice_location
                                prior_dcm.ImagePositionPatient = image_position_patient.tolist()

                                # write interpolated slice to DICOM
                                interp_dcm_file = temp_dir / f"{slice_location}.dcm"
                                pydicom.write_file(interp_dcm_file, prior_dcm)

                                # insert into dataframe
                                interp_df_row = df_files.iloc[missing_index]
                                interp_df_row["slice_location"] = slice_location
                                interp_df_row["file_path"] = str(interp_dcm_file)
                                df_files = df_files.append(interp_df_row, ignore_index=True)
                                df_files.sort_values(by="slice_location", inplace=True)

                        else:
                            raise ValueError("Slice Locations are not evenly spaced")

                    series_files = df_files.file_path.tolist()
                    series_files = [str(f) for f in series_files]
                    series = sitk.ReadImage(series_files)

                    output_file_base = f"CT_{series_uid_hash}"
                    nifti_file_name = f"{output_file_base}.nii.gz"
                    nifti_file = self.output_directory.joinpath(
                        patient_id, "images", nifti_file_name
                    )
                    nifti_file.parent.mkdir(exist_ok=True, parents=True)
                    sitk.WriteImage(series, str(nifti_file))
                    logger.debug("Writing CT Image Series to: %s", nifti_file)

                    json_file_name = f"{output_file_base}.json"
                    json_file = self.output_directory.joinpath(
                        patient_id, "images", json_file_name
                    )
                    convert_dicom_headers(series_files[0], nifti_file_name, json_file)

                elif sop_class_uid == RT_STRUCTURE_STORAGE_UID:

                    # Should only be one file per RTSTRUCT series
                    if not len(df_files) == 1:
                        ValueError("More than one RTSTRUCT with the same SeriesInstanceUID")

                    rt_struct_file = df_files.iloc[0]

                    # Get the linked image
                    linked_uid = rt_struct_file.referenced_uid
                    df_linked_series = self.df_preprocess[
                        self.df_preprocess.series_uid == rt_struct_file.referenced_uid
                    ]

                    # If not linked via referenced UID, then try to link via FOR
                    if len(df_linked_series) == 0:
                        for_uid = rt_struct_file.for_uid
                        df_linked_series = self.link_via_frame_of_reference(for_uid)

                    # Check that the linked series is available
                    # TODO handle rendering the masks even if we don't have an image series it's
                    # linked to
                    if len(df_linked_series) == 0:
                        raise ValueError("Series Referenced by RTSTRUCT not found")

                    linked_uid_hash = hash_uid(linked_uid)

                    output_dir = self.output_directory.joinpath(
                        patient_id,
                        "structures",
                        f"{series_uid_hash}_{linked_uid_hash}",
                    )
                    output_dir.mkdir(exist_ok=True, parents=True)

                    img_file_list = df_linked_series.file_path.tolist()
                    img_file_list = [str(f) for f in img_file_list]

                    convert_rtstruct(
                        img_file_list,
                        rt_struct_file.file_path,
                        prefix="",
                        output_dir=output_dir,
                        output_img=None,
                        spacing=None,
                    )

                    # TODO Make generation of NRRD file optional
                    nrrd_file = self.output_directory.joinpath(
                        patient_id,
                        "structures",
                        f"{series_uid_hash}_{linked_uid_hash}.nrrd",
                    )

                    write_nrrd_from_mask_directory(output_dir, nrrd_file)

                    # Save JSON
                    json_file_name = f"{series_uid_hash}_{linked_uid_hash}.nrrd"
                    json_file = self.output_directory.joinpath(
                        patient_id,
                        "structures",
                        f"{series_uid_hash}_{linked_uid_hash}.json",
                    )
                    convert_dicom_headers(
                        rt_struct_file.file_path, f"{series_uid_hash}_{linked_uid_hash}", json_file
                    )

                elif sop_class_uid == PET_IMAGE_STORAGE_UID:

                    series_files = df_files.file_path.tolist()
                    series_files = [str(f) for f in series_files]

                    output_file_base = f"PT_{series_uid_hash}"
                    nifti_file_name = f"{output_file_base}.nii.gz"
                    nifti_file = self.output_directory.joinpath(
                        patient_id, "images", nifti_file_name
                    )
                    nifti_file.parent.mkdir(exist_ok=True, parents=True)

                    convert_dicom_to_nifti_pt(
                        series_files,
                        nifti_file,
                    )

                    json_file_name = f"{output_file_base}.json"
                    json_file = self.output_directory.joinpath(
                        patient_id, "images", json_file_name
                    )
                    convert_dicom_headers(series_files[0], nifti_file_name, json_file)

                elif sop_class_uid == RT_PLAN_STORAGE_UID:

                    # No Nifti to create here, just save the JSON

                    # Should only be one file per RTPLAN series
                    if not len(df_files) == 1:
                        ValueError("More than one RTPLAN with the same SeriesInstanceUID")

                    rt_plan_file = df_files.iloc[0]

                    # Get the linked structure set
                    linked_uid = rt_plan_file.referenced_uid
                    df_linked_series = self.df_preprocess[
                        self.df_preprocess.sop_instance_uid == rt_plan_file.referenced_uid
                    ]

                    # If not linked via referenced UID, then try to link via FOR
                    if len(df_linked_series) == 0:
                        for_uid = rt_plan_file.for_uid
                        df_linked_series = self.link_via_frame_of_reference(for_uid)

                    # Check that the linked series is available
                    if len(df_linked_series) == 0:
                        raise ValueError("Series Referenced by RTPLAN not found")

                    linked_uid_hash = hash_uid(df_linked_series.iloc[0].series_uid)

                    output_file_base = f"RP_{series_uid_hash}_{linked_uid_hash}"
                    json_file_name = f"{output_file_base}.json"
                    json_file = self.output_directory.joinpath(patient_id, "plans", json_file_name)
                    json_file.parent.mkdir(exist_ok=True, parents=True)

                    convert_dicom_headers(rt_plan_file.file_path, "", json_file)

                elif sop_class_uid == RT_DOSE_STORAGE_UID:

                    # Should only be one file per RTDOSE series
                    if not len(df_files) == 1:
                        ValueError("More than one RTDOSE with the same SeriesInstanceUID")

                    rt_dose_file = df_files.iloc[0]

                    # Get the linked plan
                    linked_uid = rt_dose_file.referenced_uid
                    df_linked_series = self.df_preprocess[
                        self.df_preprocess.sop_instance_uid == rt_dose_file.referenced_uid
                    ]

                    # If not linked via referenced UID, then try to link via FOR
                    if len(df_linked_series) == 0:
                        for_uid = rt_dose_file.for_uid
                        df_linked_series = self.link_via_frame_of_reference(for_uid)

                    # Check that the linked series is available
                    if len(df_linked_series) == 0:
                        raise ValueError("Series Referenced by RTDOSE not found")

                    linked_uid_hash = hash_uid(df_linked_series.iloc[0].series_uid)

                    output_file_base = f"RD_{series_uid_hash}_{linked_uid_hash}"
                    nifti_file_name = f"{output_file_base}.nii.gz"
                    nifti_file = self.output_directory.joinpath(
                        patient_id, "doses", nifti_file_name
                    )
                    nifti_file.parent.mkdir(exist_ok=True, parents=True)
                    logger.debug("Writing RTDOSE to: %s", nifti_file)
                    convert_rtdose(rt_dose_file.file_path, nifti_file)

                    json_file_name = f"{output_file_base}.json"
                    json_file = self.output_directory.joinpath(patient_id, "doses", json_file_name)
                    convert_dicom_headers(rt_dose_file.file_path, nifti_file_name, json_file)
                else:
                    raise NotImplementedError(
                        "Unable to convert Series with SOP Class UID: {sop_class_uid} / "
                        f"Modality: {modality}"
                    )

            except Exception as e:  # pylint: disable=broad-except
                # Broad except ok here, since we will put these file into a
                # quarantine location for further inspection.
                logger.exception(e)
                logger.error("Unable to convert series with UID: %s", series_uid)

                for f in df_files.file_path.tolist():
                    logger.error(
                        "Error parsing file %s: %s. Placing file into Quarantine folder...", f, e
                    )
                    copy_file_to_quarantine(Path(f), self.output_directory, e)
