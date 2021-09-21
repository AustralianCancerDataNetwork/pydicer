import zipfile
import urllib.request
import tempfile

from pathlib import Path

from pydicer.input.base import InputBase


def download_and_extract_zip_file(zip_url, output_directory):

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file = Path(temp_dir).joinpath("tmp.zip")

        with urllib.request.urlopen(zip_url) as dl_file:

            with open(temp_file, "wb") as out_file:

                out_file.write(dl_file.read())

        with zipfile.ZipFile(temp_file, "r") as zip_ref:
            zip_ref.extractall(output_directory)


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

        num_files = [f for f in self.working_directory.glob("*")]
        if len(num_files) > 0:
            print("Warning: Directory not empty, won't download files")
            return

        download_and_extract_zip_file(self.data_url, self.working_directory)
