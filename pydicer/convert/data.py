import logging
import tempfile
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pydicom
from pydicer.convert.pt import convert_dicom_to_nifty_pt

from pydicer.convert.rtstruct import convert_rtstruct, write_nrrd_from_mask_directory
from pydicer.utils import hash_uid

from pydicer.constants import RT_STRUCTURE_STORAGE_UID, CT_IMAGE_STORAGE_UID, PET_IMAGE_STORAGE_UID

logger = logging.getLogger(__name__)

# TODO make this user-selected
INTERPOLATE_MISSING_DATA = True


class ConvertData:
    """
    Class that facilitates the conversion of the data into its intended final type

    Args:
        - :
        df_preprocess (pd.DataFrame): the DataFrame which was produced by PreprocessData
        output_directory (str|pathlib.Path, optional): Directory in which to store converted data.
            Defaults to ".".
    """

    def __init__(self, df_preprocess, output_directory="."):
        self.df_preprocess = df_preprocess
        self.output_directory = Path(output_directory)

    def convert(self):
        """
        Function to convert the data into its intended form (eg. images into Nifti)
        """

        for series_uid, df_files in self.df_preprocess.groupby("series_uid"):

            # Grab the patied_id, study_uid, sop_class_uid and modality (should be the same for all
            # files in series)
            patient_id = df_files.patient_id.unique()[0]
            study_uid = df_files.study_uid.unique()[0]
            sop_class_uid = df_files.sop_class_uid.unique()[0]
            modality = df_files.sop_class_uid.unique()[0]

            # Prepare some hashes of these UIDs to use for directory/file output paths
            study_id_hash = hash_uid(study_uid)
            series_uid_hash = hash_uid(series_uid)

            try:

                if sop_class_uid == CT_IMAGE_STORAGE_UID:
                    # Check that the slice location spacing is consistent, if not raise and error
                    # for now
                    slice_location_diffs = np.diff(df_files.slice_location.to_numpy())
                    unique_slice_diffs, num_unique_slice_diffs = np.unique(
                        slice_location_diffs, return_counts=True
                    )
                    expected_slice_diff = unique_slice_diffs[0]
                    if len(unique_slice_diffs) > 1:

                        logger.warning(
                            "Missing DICOM slices found: %s", df_files.iloc[0]["series_uid"]
                        )

                        temp_dir = Path(tempfile.mkdtemp())

                        # TODO Handle inconsistent slice spacing
                        if INTERPOLATE_MISSING_DATA:
                            # find where the missing slices are
                            missing_indices = np.where(
                                slice_location_diffs != expected_slice_diff
                            )[0]

                            for missing_index in missing_indices:
                                logger.debug("Interpolating missing slice %s", missing_index)
                                # locate nearest DICOM files to the missing slices
                                prior_dcm_file = df_files.iloc[missing_index]["file_path"]
                                next_dcm_file = df_files.iloc[missing_index + 1]["file_path"]

                                prior_dcm = pydicom.read_file(prior_dcm_file)
                                next_dcm = pydicom.read_file(next_dcm_file)

                                logger.debug("Read in adjacent DICOM files")

                                # TODO add other interp options (cubic)
                                interp_array = np.array(
                                    (prior_dcm.pixel_array + next_dcm.pixel_array) / 2,
                                    prior_dcm.pixel_array.dtype,
                                )

                                logger.debug("Computed missing image data")

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

                                logger.debug("Computed spatial information for missing slice")

                                # insert new spatial information into interpolated slice
                                prior_dcm.SliceLocation = slice_location
                                prior_dcm.ImagePositionPatient = image_position_patient.tolist()

                                logger.debug("Set DICOM tags")

                                # write interpolated slice to DICOM
                                interp_dcm_file = temp_dir / f"{slice_location}.dcm"
                                pydicom.write_file(interp_dcm_file, prior_dcm)

                                logger.debug("Wrote DICOM to temp file")

                                # insert into dataframe
                                interp_df_row = df_files.iloc[missing_index]
                                interp_df_row["slice_location"] = slice_location
                                interp_df_row["file_path"] = str(interp_dcm_file)
                                df_files = df_files.append(interp_df_row, ignore_index=True)
                                df_files.sort_values(by="slice_location", inplace=True)

                                logger.debug("Insert data to dataframe")

                        else:
                            raise ValueError("Slice Locations are not evenly spaced")

                    series_files = df_files.file_path.tolist()
                    series_files = [str(f) for f in series_files]
                    series = sitk.ReadImage(series_files)

                    output_file = self.output_directory.joinpath(
                        patient_id, study_id_hash, "images", f"CT_{series_uid_hash}.nii.gz"
                    )
                    output_file.parent.mkdir(exist_ok=True, parents=True)
                    sitk.WriteImage(series, str(output_file))
                    logger.debug("Writing CT Image Series to: %s", output_file)

                elif sop_class_uid == RT_STRUCTURE_STORAGE_UID:

                    continue

                    # Should only be one file per RTSTRUCT series
                    if not len(df_files) == 1:
                        ValueError("More than one RTSTRUCT with the same SeriesInstanceUID")

                    rt_struct_file = df_files.iloc[0]

                    # Get the linked image
                    # TODO Link via alternative method if referenced_series_uid is not available
                    linked_uid = rt_struct_file.referenced_series_uid
                    df_linked_series = self.df_preprocess[
                        self.df_preprocess.series_uid == rt_struct_file.referenced_series_uid
                    ]

                    # Check that the linked series is available
                    # TODO handle rendering the masks even if we don't have an image series it's
                    # linked to
                    if len(df_linked_series) == 0:
                        raise ValueError("Series Referenced by RTSTRUCT not found")

                    linked_uid_hash = hash_uid(linked_uid)

                    output_dir = self.output_directory.joinpath(
                        patient_id,
                        study_id_hash,
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

                    # TODO Make generation of NRRD file optional (especially because these files
                    # are quite large), as well as the colormap configurable
                    nrrd_file = self.output_directory.joinpath(
                        patient_id,
                        study_id_hash,
                        "structures",
                        f"{series_uid_hash}_{linked_uid_hash}.nrrd",
                    )

                    write_nrrd_from_mask_directory(output_dir, nrrd_file)
                elif sop_class_uid == PET_IMAGE_STORAGE_UID:

                    # all_files = file_dic["files"]
                    series_files = df_files.file_path.tolist()
                    series_files = [str(f) for f in series_files]

                    output_file = self.output_directory.joinpath(
                        patient_id, study_id_hash, "images", f"PT_{series_uid_hash}.nii.gz"
                    )
                    output_file.parent.mkdir(exist_ok=True, parents=True)

                    convert_dicom_to_nifty_pt(
                        series_files,
                        output_file,
                    )

                else:
                    raise NotImplementedError(
                        "Unable to convert Series with SOP Class UID: {sop_class_uid} / "
                        f"Modality: {modality}"
                    )

            except Exception as e:  # pylint: disable=broad-except
                logger.error(e)
                logger.error("Unable to convert series with UID")
