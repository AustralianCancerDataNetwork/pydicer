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


class TestInput(InputBase):
    def fetch_data(self):

        data_url = "https://zenodo.org/record/5276878/files/HNSCC.zip"

        download_and_extract_zip_file(data_url, self.working_directory)
