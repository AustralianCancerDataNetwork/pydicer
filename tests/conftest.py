import zipfile
import tempfile
import urllib
from pathlib import Path

import pytest


@pytest.fixture
def test_data_converted():
    """Fixture to grab the test data with already converted into PyDicer format"""

    zip_url = "https://zenodo.org/record/8237552/files/HNSCC_pydicer.zip"
    output_directory = Path("./pydicer_testdata")
    if not output_directory.exists():
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = Path(temp_dir).joinpath("tmp.zip")

            with urllib.request.urlopen(zip_url) as dl_file:
                with open(temp_file, "wb") as out_file:
                    out_file.write(dl_file.read())

            with zipfile.ZipFile(temp_file, "r") as zip_ref:
                zip_ref.extractall(output_directory)

    return output_directory.joinpath("testdata")
