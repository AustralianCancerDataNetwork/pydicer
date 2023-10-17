import pytest

from pydicer.utils import fetch_converted_test_data


@pytest.fixture
def test_data_converted():
    """Fixture to grab the test data with already converted into PyDicer format"""

    return fetch_converted_test_data("./testdata_hnscc", dataset="HNSCC")


@pytest.fixture
def test_data_autoseg():
    """Fixture to grab the test data in PyDicer format for auto-seg tests"""

    return fetch_converted_test_data("./testdata_lctsc", dataset="LCTSC")
