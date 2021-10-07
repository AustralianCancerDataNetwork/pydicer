import hashlib
import pickle


def hash_id(identifier):
    """
    Function to return the SHA256 hash of a DICOM UID

    Args:
        identifier (str): DICOM UID

    Returns:
        (str): resultant SHA256 hash of the UID
    """
    hash_sha = hashlib.sha256()
    hash_sha.update(identifier.encode("UTF-8"))
    id_hash = hash_sha.hexdigest()[:6]
    return id_hash


class StoreDicomAttrs:
    def __init__(
        self, patient_id, modality, study_id, series_id, linked_series_id, sop_class_id, files
    ):
        """
        Class to facilitate the storage of DICOM attributes for a particular modality

        Args:
            patient_id (str): patient id
            modality (str): series modality for which the attributes will be stored
            study_id (str): study uid
            series_id (str): series uid
            linked_series_id (str): the series uid which this series is linked to
            sop_class_uid (str): SOP class uid
            files (list): list of DICOM filepaths used for this series' nifty conversion
        """
        self.patient_id = patient_id
        self.modality = modality
        self.study_id = study_id
        self.hash_study_id = hash_id(study_id)
        self.series_id = series_id
        self.hash_series_id = hash_id(series_id)
        self.linked_series_id = linked_series_id
        self.hash_linked_series_id = hash_id(linked_series_id)
        self.sop_class_id = sop_class_id
        self.files = files

    def export_attrs(self, export_type, output_path, file_extension="json"):
        dic = {
            "patient_id": self.patient_id,
            "modality": self.modality,
            "study_uid": self.study_id,
            "hashed_study_uid": self.hash_study_id,
            "series_uid": self.series_id,
            "hashed_series_uid": self.hash_series_id,
            "linked_series_uid": self.linked_series_id,
            "hashed_linked_series_uid": self.hash_linked_series_id,
            "sop_class_id": self.sop_class_id,
            "dicom_files": self.files,
        }
        if export_type == "ct":
            filename = f"{self.modality}_{self.hash_series_id}"
        elif export_type == "struct":
            filename = f"{self.modality}_{self.hash_series_id}_{self.hash_linked_series_id}"

        if file_extension == "pickle":
            filename += ".p"
            with open(output_path.joinpath(filename), "wb") as outfile:
                pickle.dump(dic, outfile)
        else:
            print("No Attribute export method other than pickle has been implemented yet...")
