import logging

import pydicom
import numpy as np

logger = logging.getLogger(__name__)


class PreprocessData:
    """
    Class for preprocessing the data information into a dicionary that holds the data in a
    structured hierarchy

    Args:
        working_directory (Path): The working directory in which the data is stored (Output of the
        Input module)
    """

    def __init__(self, working_directory):
        self.working_directory = working_directory

    # TODO: need to find the linked series UID
    def preprocess(self):
        """
        Function to preprocess information regarding the data located in an Input working directory

        Returns: res_dict (dict): keys are series UIDs. For each series; we have 5 lower keys:
            - the hashed patient ID
            - the hashed study UID
            - a list of dicts that has 2 keys (path to file, slice location)
            - the series modality
            - the series UID that it is linked to

        Ex:
            {
                "series_uid": {
                    "patient_id": "",
                    "study_id": "",
                    "files": [],
                    "modality": "",
                    "linked_series_uid": ""
                }
            }
        """
        res_dict = {}
        files = self.working_directory.glob("**/*.dcm")

        for file in files:
            ds = pydicom.read_file(file, force=True)
            try:
                if ds.SeriesInstanceUID not in res_dict:
                    res_dict[ds.SeriesInstanceUID] = {
                        "patient_id": ds.PatientID,
                        "study_id": ds.StudyInstanceUID,
                        "files": [],
                        "modality": ds.Modality,
                        "linked_series_uid": "",
                    }

                image_position = np.array(ds.ImagePositionPatient, dtype=float)
                image_orientation = np.array(ds.ImageOrientationPatient, dtype=float)

                image_plane_normal = np.cross(image_orientation[:3], image_orientation[3:])

                slice_location = (image_position * image_plane_normal)[2]

                temp_dict = {"path": file, "slice_location": slice_location}

                res_dict[ds.SeriesInstanceUID]["files"].append(temp_dict)
            except Exception as e:  # pylint: disable=broad-except
                # Broad except ok here, since we will put these file into a
                # quarantine location for further inspection.
                logger.error("Error parsing file %s: %s", file, e)

        for _, value in res_dict.items():
            value["files"].sort(key=lambda x: x["slice_location"])

        return res_dict
