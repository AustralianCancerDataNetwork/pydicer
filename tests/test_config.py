# pylint: disable=redefined-outer-name,missing-function-docstring

import tempfile

import pytest
from pydicer.config import PyDicerConfig


def test_generate_nrrd_config():

    with tempfile.TemporaryDirectory() as directory:

        config = PyDicerConfig(directory)

        # Assert that generate NRRD is True (default)
        assert config.get_config("generate_nrrd")

        # Update the config
        config.set_config("generate_nrrd", False)

        # Assert that it is now False
        assert not config.get_config("generate_nrrd")


def test_config_not_exists():

    with tempfile.TemporaryDirectory() as directory:

        config = PyDicerConfig(directory)

        with pytest.raises(AttributeError):
            config.get_config("doesn't_exist")

        with pytest.raises(AttributeError):
            config.set_config("doesn't_exist", 123)


def test_config_invalid_value():

    with tempfile.TemporaryDirectory() as directory:

        config = PyDicerConfig(directory)

        with pytest.raises(ValueError):
            config.set_config("generate_nrrd", 123)
