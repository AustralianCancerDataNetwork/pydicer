# pylint: disable=redefined-outer-name,missing-function-docstring

import pytest

import pandas as pd
import SimpleITK as sitk

from pydicer.generate.object import add_object, add_dose_object, add_structure_object
from pydicer.utils import read_converted_data


@pytest.fixture
def test_data_path(tmp_path_factory):
    """Fixture to generate a pydicer style file structure. For the purposes of these tests, it
    doesn't really matter what the files themselves contain. Only the converted.csv will be used
    here."""

    working_directory = tmp_path_factory.mktemp("data")

    cols = [
        "",
        "sop_instance_uid",
        "hashed_uid",
        "modality",
        "patient_id",
        "series_uid",
        "for_uid",
        "referenced_sop_instance_uid",
        "path",
    ]
    rows = [
        [
            0,
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.418136430763474248173140712714",
            "b281ea",
            "CT",
            "HNSCC-01-0019",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.233510441938368266923995238976",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.290727775603409136366833989550",
            "",
            "data/HNSCC-01-0019/images/b281ea",
        ],
        [
            0,
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.168221415040968580239112565792",
            "7cdcd9",
            "RTSTRUCT",
            "HNSCC-01-0019",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.103450757970418393826743010361",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.290727775603409136366833989550",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.418136430763474248173140712714",
            "data/HNSCC-01-0019/structures/7cdcd9",
        ],
        [
            0,
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.254865609982571308239859201936",
            "57b99f",
            "RTPLAN",
            "HNSCC-01-0019",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.202542618630321306831779497186",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.290727775603409136366833989550",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.168221415040968580239112565792",
            "data/HNSCC-01-0019/plans/57b99f",
        ],
        [
            0,
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.242809596262952988524850819667",
            "309e1a",
            "RTDOSE",
            "HNSCC-01-0019",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.777975715563610987698151746284",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.290727775603409136366833989550",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.254865609982571308239859201936",
            "data/HNSCC-01-0019/doses/309e1a",
        ],
    ]

    df_converted = pd.DataFrame(rows, columns=cols)
    for _, row in df_converted.iterrows():

        data_obj_path = working_directory.joinpath(row.path)
        data_obj_path.mkdir(parents=True, exist_ok=True)

    converted_path = working_directory.joinpath("data", "HNSCC-01-0019", "converted.csv")
    df_converted.to_csv(converted_path)

    # Also create a dataset directory with converted sub-set
    dataset_path = working_directory.joinpath("test_dataset", "HNSCC-01-0019")
    dataset_path.mkdir(parents=True)
    converted_path = dataset_path.joinpath("converted.csv")
    df_converted[:2].to_csv(converted_path)

    return working_directory


def test_generate_patient_id_does_not_exist(test_data_path):

    with pytest.raises(ValueError):
        add_object(test_data_path, "test_id", "test_pat", "image", "CT")


def test_generate_incorrect_image_type(test_data_path):

    with pytest.raises(ValueError):
        add_object(test_data_path, "test_id", "HNSCC-01-0019", "oops", "CT")


def test_generate_object_does_not_exist(test_data_path):

    with pytest.raises(SystemError):
        add_object(test_data_path, "test_id", "HNSCC-01-0019", "image", "CT")


def test_generate_object_already_exists(test_data_path):

    with pytest.raises(SystemError):
        add_object(test_data_path, "b281ea", "HNSCC-01-0019", "image", "CT")


def test_generate_object(test_data_path):

    test_obj_path = test_data_path.joinpath("data", "HNSCC-01-0019", "images", "test_id")
    test_obj_path.mkdir()

    # Confirm the data object isn't there yet
    df_converted = read_converted_data(test_data_path, patients=["HNSCC-01-0019"])
    assert len(df_converted[df_converted.hashed_uid == "test_id"]) == 0

    # Add the object
    add_object(test_data_path, "test_id", "HNSCC-01-0019", "image", "CT")

    # Now make sure it's there
    df_converted = read_converted_data(test_data_path, patients=["HNSCC-01-0019"])
    assert len(df_converted[df_converted.hashed_uid == "test_id"]) == 1


def test_generate_object_add_to_dataset(test_data_path):

    test_obj_path = test_data_path.joinpath("data", "HNSCC-01-0019", "images", "test_id")
    test_obj_path.mkdir()

    # Confirm the data object isn't there yet
    df_converted = read_converted_data(
        test_data_path, dataset_name="test_dataset", patients=["HNSCC-01-0019"]
    )
    assert len(df_converted[df_converted.hashed_uid == "test_id"]) == 0

    add_object(
        test_data_path, "test_id", "HNSCC-01-0019", "image", "CT", datasets=["test_dataset"]
    )
    # Now make sure it's there
    df_converted = read_converted_data(
        test_data_path, dataset_name="test_dataset", patients=["HNSCC-01-0019"]
    )
    assert len(df_converted[df_converted.hashed_uid == "test_id"]) == 1


def test_generate_dose_object(test_data_path):

    # Confirm the data object isn't there yet
    df_converted = read_converted_data(test_data_path, patients=["HNSCC-01-0019"])
    assert len(df_converted[df_converted.hashed_uid == "dose_id"]) == 0

    test_dose = sitk.Image(20, 20, 20, sitk.sitkFloat32)
    linked_structure_hash = "7cdcd9"
    add_dose_object(test_data_path, test_dose, "dose_id", "HNSCC-01-0019", linked_structure_hash)

    # Now make sure it's there
    df_converted = read_converted_data(test_data_path, patients=["HNSCC-01-0019"])
    assert len(df_converted[df_converted.hashed_uid == "dose_id"]) == 1

    # Also make sure the for_uid and reference sop instance uid are correct
    linked_row = df_converted[df_converted.hashed_uid == "dose_id"].iloc[0]
    assert linked_row.for_uid == "1.3.6.1.4.1.14519.5.2.1.1706.8040.290727775603409136366833989550"
    assert (
        linked_row.referenced_sop_instance_uid
        == "1.3.6.1.4.1.14519.5.2.1.1706.8040.168221415040968580239112565792"
    )

    # And that the dose file exists
    assert test_data_path.joinpath(
        "data", "HNSCC-01-0019", "doses", "dose_id", "RTDOSE.nii.gz"
    ).exists()


def test_generate_structure_object(test_data_path):

    # Confirm the data object isn't there yet
    df_converted = read_converted_data(test_data_path, patients=["HNSCC-01-0019"])
    assert len(df_converted[df_converted.hashed_uid == "structure_id"]) == 0

    test_structure_set = {
        "test_struct1": sitk.Image(20, 20, 20, sitk.sitkFloat32),
        "test_struct2": sitk.Image(20, 20, 20, sitk.sitkFloat32),
    }
    linked_image_hash = "b281ea"
    add_structure_object(
        test_data_path, test_structure_set, "structure_id", "HNSCC-01-0019", linked_image_hash
    )

    # Now make sure it's there
    df_converted = read_converted_data(test_data_path, patients=["HNSCC-01-0019"])
    assert len(df_converted[df_converted.hashed_uid == "structure_id"]) == 1

    # Also make sure the for_uid and reference sop instance uid are correct
    linked_row = df_converted[df_converted.hashed_uid == "structure_id"].iloc[0]
    assert linked_row.for_uid == "1.3.6.1.4.1.14519.5.2.1.1706.8040.290727775603409136366833989550"
    assert (
        linked_row.referenced_sop_instance_uid
        == "1.3.6.1.4.1.14519.5.2.1.1706.8040.418136430763474248173140712714"
    )

    # And that the structure files actually exist
    assert test_data_path.joinpath(
        "data", "HNSCC-01-0019", "structures", "structure_id", "test_struct1.nii.gz"
    ).exists()
