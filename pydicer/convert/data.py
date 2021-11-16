import logging
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import pydicom
from pydicer.convert.pt import convert_dicom_to_nifty_pt

from pydicer.convert.rtstruct import convert_rtstruct, write_nrrd_from_mask_directory
from pydicer.utils import hash_uid

from pydicer.constants import (
    RT_STRUCTURE_STORAGE_UID,
    CT_IMAGE_STORAGE_UID,
    PET_IMAGE_STORAGE_UID
)

logger = logging.getLogger(__name__)


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
                    slice_location_diffs = np.gradient(df_files.slice_location.to_numpy())
                    if not np.allclose(slice_location_diffs, slice_location_diffs[0]):
                        # TODO Handle inconsistent slice spacing
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

                # TODO send to quarantine

                temp_dcm = pydicom.dcmwrite()
                series
                
