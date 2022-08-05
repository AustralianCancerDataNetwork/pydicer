# pylint: disable=redefined-outer-name,missing-function-docstring

from pathlib import Path

import pytest
from pydicer.config import PyDicerConfig

from pydicer.input.test import TestInput

@pytest.fixture
def test_data():
    """Fixture to grab the test data"""

    directory = Path("./testdata")
    directory.mkdir(exist_ok=True, parents=True)

    working_directory = directory.joinpath("dicom")
    working_directory.mkdir(exist_ok=True, parents=True)

    test_input = TestInput(working_directory)
    test_input.fetch_data()

    return working_directory


def test_generate_nrrd_config(test_data):

    config = PyDicerConfig(test_data)

    # Assert that generate NRRD is True (default)
    assert config.get_config("generate_nrrd")

    # Update the config
    config.set_config("generate_nrrd", False)

    # Assert that it is now False
    assert not config.get_config("generate_nrrd")

def test_config_not_exists(test_data):

    config = PyDicerConfig(test_data)

    with pytest.raises(AttributeError):
        config.get_config("doesn't_exist")

    with pytest.raises(AttributeError):
        config.set_config("doesn't_exist", 123)

def test_config_invalid_value(test_data):

    config =  PyDicerConfig(test_data)

    with pytest.raises(ValueError):
        config.set_config("generate_nrrd", 123)
