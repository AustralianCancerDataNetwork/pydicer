# pylint: disable=redefined-outer-name,missing-function-docstring

import os
import logging
import shutil
import json

from pathlib import Path
import SimpleITK as sitk
import pytest

from pydicer import PyDicer
from pydicer.dataset.nnunet import NNUNetDataset
from pydicer.utils import add_structure_name_mapping


def test_nnunet_env_error(test_data_autoseg):
    if "nnUNet_raw_data_base" in os.environ:
        del os.environ["nnUNet_raw_data_base"]

    working_directory = test_data_autoseg

    # Expect SystemError due to nnUNet_raw_data_base not being set
    with pytest.raises(SystemError):
        NNUNetDataset(
            working_directory=working_directory,
            nnunet_id=100,
            nnunet_name="TestTask",
            nnunet_description="A test nnUNet task.",
        )


def test_nnunet_env_ok(test_data_autoseg):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg

    # With nnUNet_raw_data_base this should success without exception
    NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
    )


def test_nnunet_check_dataset_ok(test_data_autoseg):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
    )

    # This dataset should be ready to go for nnUNet so no exception here
    nnunet.check_dataset()


def test_nnunet_check_dataset_not_ok(test_data_autoseg):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg
    pyd = PyDicer(working_directory)

    # Prepare a subset of data we know if invalid for the purpose of this test
    def pick_ct_only(df):
        return df[df.modality == "CT"]

    dataset_name = "rubbish"
    pyd.dataset.prepare(dataset_name, pick_ct_only)

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
        dataset_name=dataset_name,
    )

    # This should raise an error now since our dataset is invalid
    with pytest.raises(SystemError):
        nnunet.check_dataset()


def test_nnunet_split_dataset_random(test_data_autoseg):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
    )

    nnunet.split_dataset(random_state=42)

    assert "LCTSC-Train-S1-007" in nnunet.training_cases
    assert "LCTSC-Train-S1-002" in nnunet.training_cases
    assert "LCTSC-Train-S1-006" in nnunet.training_cases
    assert "LCTSC-Test-S1-101" in nnunet.training_cases
    assert "LCTSC-Test-S1-102" in nnunet.training_cases
    assert "LCTSC-Train-S1-001" in nnunet.training_cases
    assert "LCTSC-Train-S1-003" in nnunet.training_cases
    assert "LCTSC-Train-S1-005" in nnunet.testing_cases
    assert "LCTSC-Train-S1-008" in nnunet.testing_cases
    assert "LCTSC-Train-S1-004" in nnunet.testing_cases


def test_nnunet_split_dataset_specify(test_data_autoseg):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
    )

    nnunet.split_dataset(
        training_cases=["LCTSC-Train-S1-007", "LCTSC-Train-S1-002"],
        testing_cases=["LCTSC-Test-S1-101"],
    )

    assert "LCTSC-Train-S1-007" in nnunet.training_cases
    assert "LCTSC-Train-S1-002" in nnunet.training_cases
    assert "LCTSC-Test-S1-101" in nnunet.testing_cases


def test_nnunet_split_dataset_invalid(test_data_autoseg):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
    )

    with pytest.raises(ValueError):
        nnunet.split_dataset(training_cases=["invalid_id1", "invalid_2"])


def test_nnunet_add_testing_case(test_data_autoseg):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
    )

    # nnunet.add_testing_cases(testing_cases==["invalid_id1", "invalid_2"])
    nnunet.add_testing_cases(testing_cases=["LCTSC-Test-S1-101"])

    assert "LCTSC-Test-S1-101" in nnunet.testing_cases


def test_nnunet_add_testing_case_invalid(test_data_autoseg):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
    )

    with pytest.raises(ValueError):
        nnunet.add_testing_cases(testing_cases=["invalid_id1"])


def test_nnunet_check_duplicates_ok(test_data_autoseg, caplog):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg

    caplog.set_level(logging.INFO)

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
    )

    nnunet.split_dataset(
        training_cases=["LCTSC-Train-S1-007", "LCTSC-Train-S1-002"],
        testing_cases=["LCTSC-Test-S1-101"],
    )

    nnunet.check_duplicates_train_test()

    assert "No duplicate images found" in caplog.text


def test_nnunet_check_duplicates_dup(test_data_autoseg, caplog):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg

    caplog.set_level(logging.INFO)

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
    )

    nnunet.split_dataset(
        training_cases=["LCTSC-Train-S1-007", "LCTSC-Train-S1-002"],
        testing_cases=["LCTSC-Train-S1-007"],
    )

    nnunet.check_duplicates_train_test()

    assert "is likely a duplicate of" in caplog.text


def test_nnunet_check_structure_names_missing(test_data_autoseg):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg

    mapping = {
        "Heart": [],
        "Lung_L": [],
        "Lung_R": [],
    }
    mapping_id = "no_mapping"
    add_structure_name_mapping(
        mapping_dict=mapping, mapping_id=mapping_id, working_directory=working_directory
    )

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
        mapping_id=mapping_id,
    )

    nnunet.split_dataset(
        training_cases=["LCTSC-Train-S1-007", "LCTSC-Train-S1-002"],
        testing_cases=["LCTSC-Test-S1-101"],
    )

    df_results = nnunet.check_structure_names()

    assert df_results.data["Heart"].sum() == 3
    assert df_results.data["Lung_L"].sum() == 2
    assert df_results.data["Lung_R"].sum() == 2


def test_nnunet_check_structure_names_mapped(test_data_autoseg):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg

    mapping = {
        "Heart": [],
        "Lung_L": ["Lung_Left"],
        "Lung_R": ["Lung_Right"],
    }
    mapping_id = "mapping_ok"
    add_structure_name_mapping(
        mapping_dict=mapping, mapping_id=mapping_id, working_directory=working_directory
    )

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
        mapping_id=mapping_id,
    )

    nnunet.split_dataset(
        training_cases=["LCTSC-Train-S1-007", "LCTSC-Train-S1-002"],
        testing_cases=["LCTSC-Test-S1-101"],
    )

    df_results = nnunet.check_structure_names()

    assert df_results.data["Heart"].sum() == 3
    assert df_results.data["Lung_L"].sum() == 3
    assert df_results.data["Lung_R"].sum() == 3


def test_nnunet_check_overlapping_structures_ok(test_data_autoseg, caplog):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg
    caplog.set_level(logging.INFO)

    mapping = {
        "Lung_L": ["Lung_Left"],
        "Lung_R": ["Lung_Right"],
    }
    mapping_id = "mapping_ok"
    add_structure_name_mapping(
        mapping_dict=mapping, mapping_id=mapping_id, working_directory=working_directory
    )

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
        mapping_id=mapping_id,
    )

    nnunet.split_dataset(
        training_cases=["LCTSC-Train-S1-007"],
        testing_cases=["LCTSC-Test-S1-101"],
    )

    nnunet.check_structure_names()

    nnunet.check_overlapping_structures()

    assert "No overlapping structures detected" in caplog.text


def test_nnunet_check_overlapping_structures_overlap(test_data_autoseg, caplog):
    os.environ["nnUNet_raw_data_base"] = "."

    working_directory = test_data_autoseg
    caplog.set_level(logging.INFO)

    mapping = {
        "Heart": [],
        "Lung_L": ["Lung_Left"],
        "Lung_R": ["Lung_Right"],
    }
    mapping_id = "mapping_ok"
    add_structure_name_mapping(
        mapping_dict=mapping, mapping_id=mapping_id, working_directory=working_directory
    )

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
        mapping_id=mapping_id,
    )

    nnunet.split_dataset(
        training_cases=["LCTSC-Train-S1-007", "LCTSC-Train-S1-002"],
        testing_cases=["LCTSC-Test-S1-101"],
    )

    nnunet.check_structure_names()

    nnunet.check_overlapping_structures()

    assert "Overlapping structures were detected" in caplog.text


def test_nnunet_prepare_dataset(test_data_autoseg):
    raw_path = Path("./testdata_nnunet")
    os.environ["nnUNet_raw_data_base"] = str(raw_path)

    # Remove raw path if it was left over from a previous test
    if raw_path.exists():
        shutil.rmtree(raw_path)

    raw_path.mkdir()

    working_directory = test_data_autoseg

    mapping = {
        "Heart": [],
        "Lung_L": ["Lung_Left"],
        "Lung_R": ["Lung_Right"],
    }
    mapping_id = "mapping_ok"
    add_structure_name_mapping(
        mapping_dict=mapping, mapping_id=mapping_id, working_directory=working_directory
    )

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
        mapping_id=mapping_id,
    )

    nnunet.split_dataset(
        training_cases=["LCTSC-Train-S1-007", "LCTSC-Train-S1-002"],
        testing_cases=["LCTSC-Test-S1-101"],
    )

    nnunet.prepare_dataset()

    # Check the folders have been created
    task_path = raw_path.joinpath("nnUNet_raw_data", "Task100_TestTask")
    assert task_path.exists()
    assert task_path.joinpath("imagesTr").exists()
    assert task_path.joinpath("imagesTs").exists()
    assert task_path.joinpath("labelsTr").exists()
    assert task_path.joinpath("labelsTs").exists()

    # CHeck the dataset flle
    dataset_file = task_path.joinpath("dataset.json")
    assert dataset_file.exists()

    with open(dataset_file, "r", encoding="utf-8") as fp:
        ds = json.load(fp)
    assert len(ds.keys()) == 12
    assert ds["labels"] == {"0": "background", "1": "Heart", "2": "Lung_L", "3": "Lung_R"}

    # Open a label map file for sanity checks
    label_map_path = task_path.joinpath("labelsTr", "LCTSC-Train-S1-007.nii.gz")
    assert label_map_path.exists()

    label_map = sitk.ReadImage(str(label_map_path))
    lsif = sitk.LabelStatisticsImageFilter()
    lsif.Execute(label_map, label_map)
    labels = lsif.GetLabels()
    assert len(labels) == 4
    assert 0 in labels
    assert 1 in labels
    assert 2 in labels
    assert 3 in labels

    assert lsif.GetCount(1) == 24336
    assert lsif.GetCount(2) == 52307
    assert lsif.GetCount(3) == 67702


def test_nnunet_generate_training_scripts(test_data_autoseg):
    raw_path = Path("./testdata_nnunet")
    os.environ["nnUNet_raw_data_base"] = str(raw_path)

    # Remove raw path if it was left over from a previous test
    if raw_path.exists():
        shutil.rmtree(raw_path)

    raw_path.mkdir()

    working_directory = test_data_autoseg

    mapping = {
        "Heart": [],
        "Lung_L": ["Lung_Left"],
        "Lung_R": ["Lung_Right"],
    }
    mapping_id = "mapping_ok"
    add_structure_name_mapping(
        mapping_dict=mapping, mapping_id=mapping_id, working_directory=working_directory
    )

    nnunet = NNUNetDataset(
        working_directory=working_directory,
        nnunet_id=100,
        nnunet_name="TestTask",
        nnunet_description="A test nnUNet task.",
        mapping_id=mapping_id,
    )

    nnunet.split_dataset(
        training_cases=["LCTSC-Train-S1-007", "LCTSC-Train-S1-002"],
        testing_cases=["LCTSC-Test-S1-101"],
    )

    nnunet.prepare_dataset()

    nnunet.generate_training_scripts(raw_path)

    script_file = raw_path.joinpath("train_100_TestTask.sh")
    assert script_file.exists()

    with open(script_file, "r", encoding="utf-8") as fp:
        script_contents = fp.read()

    assert "#!/bin/bash" in script_contents
    assert "nnUNet_plan_and_preprocess -t 100 --verify_dataset_integrity;" in script_contents
    assert "nnUNet_train 2d nnUNetTrainerV2 Task100_TestTask all;" in script_contents
