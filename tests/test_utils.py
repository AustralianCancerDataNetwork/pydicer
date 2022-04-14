# pylint: disable=redefined-outer-name,missing-function-docstring

from pathlib import Path

import pytest
import pydicom

from pydicer.input.test import TestInput
from pydicer.utils import determine_dcm_datetime


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


def test_fetch_ds_datetime(test_data):

    rt_struct_file = test_data.joinpath(
        "HNSCC",
        "HNSCC-01-0019",
        "07-04-1998-NA-RT SIMULATION-48452",
        "1.000000-NA-10361",
        "1-1.dcm",
    )

    ds = pydicom.read_file(rt_struct_file)
    ds_datetime = determine_dcm_datetime(ds)
    assert ds_datetime.year == 1998
    assert ds_datetime.month == 7
    assert ds_datetime.day == 4
    assert ds_datetime.hour == 0
    assert ds_datetime.minute == 0

    ds_datetime = determine_dcm_datetime(ds, require_time=True)
    assert ds_datetime.year == 2001
    assert ds_datetime.month == 10
    assert ds_datetime.day == 28
    assert ds_datetime.hour == 12
    assert ds_datetime.minute == 48
