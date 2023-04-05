# pylint: disable=redefined-outer-name,missing-function-docstring
import json
from pathlib import Path

import pytest

from pydicer.input.test import TestInput
from pydicer.dataset.preparation import MapStructureSetNomenclature


@pytest.fixture
def test_data():
    """Fixture to grab the test data"""

    directory = Path("./testdata")
    directory.mkdir(exist_ok=True, parents=True)

    working_directory = directory.joinpath("dicom")
    working_directory.mkdir(exist_ok=True, parents=True)

    test_input = TestInput(working_directory)
    test_input.fetch_data()

    return directory


def test_map_single_structure_set(test_data):

    patient_id = "HNSCC-01-0199"
    struct_set_id = "06e49c"

    # Create mapping file for this structure set and write to "struct_set_id" directory
    mapping_dict = {"structures": {"AVVVOID": ["avoid"]}}

    struct_set_path = (
        test_data.joinpath("data")
        .joinpath(patient_id)
        .joinpath("structures")
        .joinpath(struct_set_id)
    )

    mapping_file = struct_set_path.joinpath("structures_map.json")
    structure_name = struct_set_path.joinpath("avoid.nii.gz")
    mapped_structure_name = struct_set_path.joinpath("AVVVOID.nii.gz")
    with mapping_file.open("w", encoding="utf-8") as f:
        json.dump(mapping_dict, f)

    stan = MapStructureSetNomenclature(test_data)
    stan.map_specific_structure_set_names(struct_set_id)

    # Assert mapping has been done correctly
    assert mapped_structure_name.is_file()
    assert not structure_name.is_file()

    # Remove mapping file for this structure set
    mapping_file.unlink()
