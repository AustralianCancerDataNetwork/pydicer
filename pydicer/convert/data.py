from pathlib import Path
import SimpleITK as sitk

from pydicer.convert.rtstruct import convert_rtstruct, write_nrrd_from_mask_directory
from pydicer.convert.store_attrs import StoreDicomAttrs

from pydicer.constants import (
    RT_STRUCTURE_STORAGE_UID,
    CT_IMAGE_STORAGE_UID,
)


class ConvertData:
    """
    Class that facilitates the conversion of the data into its intended final type

    Args:
        - preprocess_dic: the dictionary that contains the preprocessed data information
    """

    def __init__(self, preprocess_dic, output_directory="."):
        self.preprocess_dic = preprocess_dic
        self.output_directory = Path(output_directory)

    def convert(self):
        """
        Function to convert the data into its intended form (eg. images into Nifti)
        """

        for series_uid, file_dic in self.preprocess_dic.items():

            series_attrs = StoreDicomAttrs(
                file_dic["patient_id"],
                file_dic["modality"],
                file_dic["study_id"],
                series_uid,
                file_dic["linked_series_uid"]["referenced_series_uid"],
                file_dic["sop_class_uid"],
                file_dic["files"],
            )

            if series_attrs.sop_class_id == CT_IMAGE_STORAGE_UID:
                series_files = [str(x["path"]) for x in series_attrs.files]
                series = sitk.ReadImage(series_files)

                output_file = self.output_directory.joinpath(
                    series_attrs.patient_id,
                    series_attrs.hash_study_id,
                    "images",
                    f"CT_{series_attrs.hash_series_id}.nii.gz",
                )
                output_file.parent.mkdir(exist_ok=True, parents=True)
                sitk.WriteImage(series, str(output_file))

                output_path = self.output_directory.joinpath(
                    series_attrs.patient_id,
                    series_attrs.hash_study_id,
                    "images"
                )
                export_type = "ct"
                
                # Save the series attibutes to file
                series_attrs.export_attrs(export_type, output_path)

            elif series_attrs.sop_class_id == RT_STRUCTURE_STORAGE_UID:

                # Get the linked image
                linked_uid = series_attrs.linked_series_id
                linked_dicom_dict = self.preprocess_dic[linked_uid]

                output_dir = self.output_directory.joinpath(
                    series_attrs.patient_id,
                    series_attrs.hash_study_id,
                    "structures",
                    f"{series_attrs.hash_series_id}_{series_attrs.hash_linked_series_id}",
                )
                output_dir.mkdir(exist_ok=True, parents=True)

                img_file_list = [str(f["path"]) for f in linked_dicom_dict["files"]]

                convert_rtstruct(
                    img_file_list,
                    series_attrs.files[0],
                    prefix="",
                    output_dir=output_dir,
                    output_img=None,
                    spacing=None,
                )

                output_path = self.output_directory.joinpath(
                    series_attrs.patient_id,
                    series_attrs.hash_study_id,
                    "structures",
                )
                export_type = "struct"

                # TODO Make generation of NRRD file optional, as well as the colormap configurable
                nrrd_file = self.output_directory.joinpath(
                    series_attrs.patient_id,
                    series_attrs.hash_study_id,
                    "structures",
                    f"{series_attrs.hash_series_id}_{series_attrs.hash_linked_series_id}.nrrd",
                )
                write_nrrd_from_mask_directory(output_dir, nrrd_file)
            
                # Save the series attibutes to file
                series_attrs.export_attrs(export_type, output_path)

