import hashlib
from pathlib import Path
import SimpleITK as sitk

from pydicer.convert.rtstruct import convert_rtstruct

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
        hash_sha = hashlib.sha256()

        for series_uid, file_dic in self.preprocess_dic.items():

            hash_sha.update(file_dic["study_id"].encode("UTF-8"))
            study_id_hash = hash_sha.hexdigest()[:6]

            hash_sha.update(series_uid.encode("UTF-8"))
            series_uid_hash = hash_sha.hexdigest()[:6]

            if file_dic["sop_class_uid"] == CT_IMAGE_STORAGE_UID:
                series_files = [str(x["path"]) for x in file_dic["files"]]
                series = sitk.ReadImage(series_files)

                output_dir = self.output_directory.joinpath(
                    file_dic["patient_id"], study_id_hash, "images", f"CT_{series_uid_hash}.nii.gz"
                )
                output_dir.parent.mkdir(exist_ok=True, parents=True)
                sitk.WriteImage(series, str(output_dir))

            elif file_dic["sop_class_uid"] == RT_STRUCTURE_STORAGE_UID:

                # Get the linked image
                linked_uid = file_dic["linked_series_uid"]["referenced_series_uid"]
                linked_dicom_dict = self.preprocess_dic[linked_uid]

                hash_sha.update(linked_uid.encode("UTF-8"))
                linked_uid_hash = hash_sha.hexdigest()[:6]

                output_dir = self.output_directory.joinpath(
                    file_dic["patient_id"], study_id_hash, "structures"
                )

                convert_rtstruct(
                    [i["path"] for i in linked_dicom_dict["files"]],
                    file_dic["files"][0],
                    prefix=f"STRUCT_{series_uid_hash}_{linked_uid_hash}",
                    output_dir=output_dir,
                    output_img=None,
                    spacing=None,
                )
