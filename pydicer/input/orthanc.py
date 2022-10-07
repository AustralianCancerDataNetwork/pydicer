import logging
from io import BytesIO

import pydicom
from pyorthanc.deprecated.client import Orthanc

from pydicer.utils import get_iterator
from pydicer.input.base import InputBase

logger = logging.getLogger(__name__)


def adapt_dataset_from_bytes(blob):
    """Convert bytes coming from Orthanc to DICOM dataset

    Args:
        blob (bytes): The bytes to convert

    Returns:
        pydicom.Dataset: The DICOM dataset
    """
    dataset = pydicom.dcmread(BytesIO(blob))
    return dataset


class OrthancInput(InputBase):
    def __init__(self, host, port, username=None, password=None, working_directory=None):
        """Class for fetching files from Orthanc.

        Args:
            host (str): The IP address or host name of the Orthanc.
            port (int): The port to use to communicate on.
            username (str, optional): Orthanc username.
            password (str, optional): Orthanc password.
            working_directory (str|pathlib.Path, optional): The working directory in which to
            store the data fetched. Defaults to a temp directory.

        Raises:
            ConnectionError: Raises a connection error if unable to verify the connection to
                Orthanc.
        """

        super().__init__(working_directory)

        if not host.startswith("http"):
            host = f"http://{host}"

        self.orthanc = Orthanc(f"{host}:{port}")

        if username is not None and password is not None:
            self.orthanc.setup_credentials(username, password)

        # Do a dummy lookup to check that we can reach the Orthanc host, this will throw a
        # connection error if we can't connect to the Orthanc
        self.orthanc.c_find({"Level": "Patient", "Query": {"PatientID": "XXX"}})

    def fetch_data(self, patients, modalities=None):
        """Download the DICOM data from Orthanc

        Args:
            patients (list|str): A list of patient IDs, or a single patient ID.
            modalities (list|str, optional): List of modalities or a single modality to fetch.
                Defaults to None where all modalities would be fetched.
        """

        if not isinstance(patients, list) and not isinstance(patients, tuple):
            patients = [patients]

        if (
            modalities is not None
            and not isinstance(modalities, list)
            and not isinstance(modalities, tuple)
        ):
            modalities = [modalities]

        for patient in get_iterator(patients, unit="patients", name="Orthanc Fetch"):

            # Find the Orthanc ID for this patient
            orthanc_patient_ids = self.orthanc.c_find(
                {"Level": "Patient", "Query": {"PatientID": patient}}
            )

            if len(orthanc_patient_ids) == 0:
                logger.warning("Patient not found in Orthanc: %s", patient)
                continue

            if len(orthanc_patient_ids) > 1:
                logger.warning(
                    "Patient returned multple Orthanc IDs: %s. Selecting first only", patient
                )

            orthanc_patient_id = orthanc_patient_ids[0]

            patient_information = self.orthanc.get_patient_information(orthanc_patient_id)
            patient_id = patient_information["MainDicomTags"]["PatientID"]

            # Loop over each study for this patient
            study_identifiers = patient_information["Studies"]
            for study_identifier in study_identifiers:

                # Loop over each series in this study
                study_information = self.orthanc.get_study_information(study_identifier)
                series_identifiers = study_information["Series"]
                for series_identifier in series_identifiers:
                    series_information = self.orthanc.get_series_information(series_identifier)

                    # Skip if this isn't one of the modalities we want
                    modality = series_information["MainDicomTags"]["Modality"]
                    if modalities is not None and not modality in modalities:
                        continue

                    series_information = self.orthanc.get_series_information(series_identifier)
                    series_instance_uid = series_information["MainDicomTags"]["SeriesInstanceUID"]

                    # Create the output directory for this series
                    series_path = self.working_directory.joinpath(patient_id, series_instance_uid)
                    series_path.mkdir(exist_ok=True, parents=True)

                    # Loop over each instance in this series
                    instance_identifiers = series_information["Instances"]
                    for instance_identifier in instance_identifiers:
                        instance_information = self.orthanc.get_instance_information(
                            instance_identifier
                        )

                        # Download the DICOM instance
                        f = self.orthanc.get_instance_file(instance_identifier)
                        ds = adapt_dataset_from_bytes(f)

                        sop_instance_uid = instance_information["MainDicomTags"]["SOPInstanceUID"]
                        ds_file_name = f"{modality}.{sop_instance_uid}.dcm"
                        ds_path = series_path.joinpath(ds_file_name)

                        # Save the DICOM dataset
                        ds.save_as(ds_path)
                        logger.debug("Saving DICOM dataset to %s", ds_path)
