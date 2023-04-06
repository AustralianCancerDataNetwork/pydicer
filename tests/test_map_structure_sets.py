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


def test_map_project_structure_set(test_data):

    # Create mapping file for the project and write to ".pydicer" directory
    mapping_dict = {"structures": {"BRAINSTEM": ["brainstem", "Brainstem"]}}

    pat_0019_struct_set = (
        test_data.joinpath("data")
        .joinpath("HNSCC-01-0019")
        .joinpath("structures")
        .joinpath("7cdcd9")
    )

    pat_0176_struct_set = (
        test_data.joinpath("data")
        .joinpath("HNSCC-01-0176")
        .joinpath("structures")
        .joinpath("cbbf5b")
    )

    pat_0199_struct_set = (
        test_data.joinpath("data")
        .joinpath("HNSCC-01-0199")
        .joinpath("structures")
        .joinpath("06e49c")
    )

    correct_mappings = {
        "patients": [
            {
                "patient_id": "HNSCC-01-0019",
                "mappings": {
                    "filepaths": {
                        pat_0019_struct_set.joinpath(
                            "Brainstem.nii.gz"
                        ): pat_0019_struct_set.joinpath("BRAINSTEM.nii.gz")
                    }
                },
            },
            {
                "patient_id": "HNSCC-01-0176",
                "mappings": {
                    "filepaths": {
                        pat_0176_struct_set.joinpath(
                            "brainstem.nii.gz"
                        ): pat_0176_struct_set.joinpath("BRAINSTEM.nii.gz")
                    },
                },
            },
            {
                "patient_id": "HNSCC-01-0199",
                "mappings": {
                    "filepaths": {
                        pat_0199_struct_set.joinpath(
                            "Brainstem.nii.gz"
                        ): pat_0199_struct_set.joinpath("BRAINSTEM.nii.gz")
                    }
                },
            },
        ]
    }

    mapping_file = test_data.joinpath(".pydicer").joinpath("structures_map.json")

    with mapping_file.open("w", encoding="utf-8") as f:
        json.dump(mapping_dict, f)

    stan = MapStructureSetNomenclature(test_data)
    stan.map_project_structure_set_names()

    # Assert new brainstem niftis are now there
    assert list(correct_mappings["patients"][0]["mappings"]["filepaths"].values())[0].is_file()
    assert list(correct_mappings["patients"][1]["mappings"]["filepaths"].values())[0].is_file()
    assert list(correct_mappings["patients"][2]["mappings"]["filepaths"].values())[0].is_file()
