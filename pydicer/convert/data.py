import pydicom
import SimpleITK as sitk
from platipy.dicom.io.crawl import safe_sort_dicom_image_list
from pathlib import Path
import hashlib


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

        return: void
        """
        m = hashlib.sha256()

        for key, value in self.preprocess_dic.items():
            if value["modality"] == "CT":
                series_files = [str(x["path"]) for x in value["files"]]
                series = sitk.ReadImage(series_files)

                m.update(value["study_id"].encode("UTF-8"))
                study_id_hash = m.hexdigest()[:6]

                m.update(key.encode("UTF-8"))
                series_uid_hash = m.hexdigest()[:6]

                output_dir = self.output_directory.joinpath(
                    value["patient_id"], study_id_hash, "images", f"CT_{series_uid_hash}.nii.gz"
                )
                output_dir.parent.mkdir(exist_ok=True, parents=True)
                sitk.WriteImage(series, str(output_dir))

        return
