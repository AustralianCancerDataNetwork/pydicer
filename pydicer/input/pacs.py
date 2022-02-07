import os
import pydicom

from platipy.dicom.communication.connector import DicomConnector

from pydicer.input.base import InputBase


class DICOMPACSInput(InputBase):
    def __init__(self, host, port, ae_title=None, working_directory=None):
        """Class for fetching files from DICOM PACS. Currently only supports C-GET commands to
        fetch the data.

        Args:
            host (str): The IP address of host name of DICOM PACS.
            port (int): The port to use to communicate on.
            ae_title (str, optional): AE Title to provide the DICOM service.
            working_directory (str|pathlib.Path, optional): The working directory in which to
            store the data fetched. Defaults to a temp directory.

        Raises:
            ConnectionError: Raises a connection error if unable to verify the connection to the
                PACS.
        """

        super().__init__(working_directory)

        self.dicom_connector = DicomConnector(
            host=host, port=port, ae_title=ae_title, output_directory=self.working_directory
        )

        if not self.dicom_connector.verify():
            raise ConnectionError("Unable to connect to DICOM PACS.")

    def fetch_data(self, patients, modalities=None):
        """Download the DICOM data from the PACS.

        Args:
            patients (list|str): A list of patient IDs, or a single patient ID. Wildcard matching
                based on the DICOM standard is supported.
            modalities (list|str, optional): List of modalities or a single modality to fetch.
                Defaults to None where all modalities would be fetched.
        """

        if not isinstance(patients, list) and not isinstance(patients, tuple):
            patients = [patients]

        if modalities is None:
            modalities = [""]

        if not isinstance(modalities, list) and not isinstance(modalities, tuple):
            modalities = [modalities]

        for patient in patients:

            dataset = pydicom.Dataset()
            dataset.PatientID = patient
            dataset.PatientName = ""
            dataset.StudyInstanceUID = ""
            dataset.ModalitiesInStudy = ""
            dataset.QueryRetrieveLevel = "STUDY"

            studies = self.dicom_connector.do_find(dataset)

            for study in studies:
                if not study:
                    continue  # These lists often contain a None study, so just skip that

                for modality in modalities:
                    dataset = pydicom.Dataset()
                    dataset.PatientID = patient
                    dataset.StudyInstanceUID = study.StudyInstanceUID
                    dataset.Modality = modality
                    dataset.SeriesInstanceUID = ""
                    dataset.QueryRetrieveLevel = "SERIES"

                    series = self.dicom_connector.do_find(dataset)
                    for s in series:
                        if not s:
                            continue  # Again, safe to skip this if None

                        if not s.PatientID == patient:
                            continue

                        # Download the series
                        self.dicom_connector.download_series(s.SeriesInstanceUID)

        # Finally, just make sure all files end with the .dcm extension
        for f in self.working_directory.glob("**/*"):
            if f.is_dir():
                continue

            if f.name.endswith(".dcm"):
                continue

            target = f.parent.joinpath(f"{f.name}.dcm")
            os.rename(f, target)
