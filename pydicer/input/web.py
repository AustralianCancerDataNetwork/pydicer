import logging

from pydicer.input.base import InputBase
from pydicer.utils import download_and_extract_zip_file

logger = logging.getLogger(__name__)


class WebInput(InputBase):
    def __init__(self, data_url, working_directory=None):
        """
        Class for downloading and saving input data off the internet

        Args:
            data_url (str): The URL of where the data is stored. For now, it must be a link to a
            zip file
            working_directory (str|pathlib.Path, optional): The working directory in which to
            store the data fetched. Defaults to a temp directory.
        """
        super().__init__(working_directory)
        self.data_url = data_url

    def fetch_data(self):
        """Download the data."""

        files_in_directory = list(self.working_directory.glob("*"))
        if len(files_in_directory) > 0:
            logger.warning("Directory not empty, won't download files")
            return

        logger.info("Downloading files from %s", self.data_url)
        download_and_extract_zip_file(self.data_url, self.working_directory)
