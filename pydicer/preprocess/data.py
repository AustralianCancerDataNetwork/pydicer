import pydicom
import numpy as np

from pydicer.constants import (
    PET_IMAGE_STORAGE_UID,
    RT_STRUCTURE_STORAGE_UID,
    CT_IMAGE_STORAGE_UID,
)


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
                    "linked_series_uid": {}
                }
            }
        """
        res_dict = {}
        files = self.working_directory.glob("**/*.dcm")

        for file in files:
            ds = pydicom.read_file(file, force=True)

            linked_series_uid = {}

            try:

                dicom_type_uid = ds.SOPClassUID

                if ds.SeriesInstanceUID not in res_dict:
                    res_dict[ds.SeriesInstanceUID] = {
                        "patient_id": ds.PatientID,
                        "study_id": ds.StudyInstanceUID,
                        "files": [],
                        "modality": ds.Modality,
                        "sop_class_uid": dicom_type_uid,
                    }

                if dicom_type_uid == RT_STRUCTURE_STORAGE_UID:

                    try:
                        referenced_series_uid = (
                            ds.ReferencedFrameOfReferenceSequence[0]
                            .RTReferencedStudySequence[0]
                            .RTReferencedSeriesSequence[0]
                            .SeriesInstanceUID
                        )
                        linked_series_uid["referenced_series_uid"] = referenced_series_uid
                    except AttributeError:
                        # Check other tags for a linked DICOM
                        # e.g. ds.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
                        # Potentially, we should check each referenced
                        referenced_frame_of_reference_uid = ds.ReferencedFrameOfReferenceSequence[
                            0
                        ].FrameOfReferenceUID
                        linked_series_uid[
                            "referenced_frame_of_reference_uid"
                        ] = referenced_frame_of_reference_uid

                    res_dict[ds.SeriesInstanceUID]["files"].append(file)

                elif dicom_type_uid in (CT_IMAGE_STORAGE_UID, PET_IMAGE_STORAGE_UID):

                    image_position = np.array(ds.ImagePositionPatient, dtype=float)
                    image_orientation = np.array(ds.ImageOrientationPatient, dtype=float)

                    image_plane_normal = np.cross(image_orientation[:3], image_orientation[3:])

                    slice_location = (image_position * image_plane_normal)[2]

                    temp_dict = {"path": file, "slice_location": slice_location}

                    res_dict[ds.SeriesInstanceUID]["files"].append(temp_dict)

                else:
                    raise ValueError(
                        f"Could not determine DICOM type {ds.Modality} {dicom_type_uid}."
                    )

            except Exception as e:  # pylint: disable=broad-except
                # Broad except ok here, since we will put these file into a
                # quarantine location for further inspection.
                print(e)

            # Include any linked DICOM series
            # This is a dictionary holding potential matching series
            res_dict[ds.SeriesInstanceUID]["linked_series_uid"] = linked_series_uid

        # Sort the files for each series by the slice_location (if available)
        for _, value in res_dict.items():

            if len(value["files"]) == 0:
                continue

            if not isinstance(value["files"][0], dict):
                continue

            if "slice_location" in value["files"][0]:
                value["files"].sort(key=lambda x: x["slice_location"])

        return res_dict
