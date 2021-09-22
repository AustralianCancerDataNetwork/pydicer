from platipy.dicom.download import tcia

from pydicer.input.base import InputBase


class TCIAInput(InputBase):
    def __init__(self, collection, patient_ids, modalities=None, working_directory=None):
        """
        Input class that interfaces with the TCIA API

        Args:
            collection (str): The TCIA collection to fetch from
            patient_ids (list, optional): The patient IDs to fetch. If not set all patients are
                fetched
            modalities (list, optional): A list of strings defining the modalites to fetch. Will
                                        fetch all modalities available if not specified.
            working_directory (str): (str|pathlib.Path, optional): The working directory in which
                to store the data fetched. Defaults to a temp directory.
        """
        super().__init__(working_directory)
        self.collection = collection
        self.patient_ids = patient_ids
        self.modalities = modalities

    def fetch_data(self):
        """
        Function to download the data from TCIA and write locally
        """

        tcia.fetch_data(
            self.collection,
            self.patient_ids,
            self.modalities,
            nifti=False,
            output_directory=self.working_directory,
        )
