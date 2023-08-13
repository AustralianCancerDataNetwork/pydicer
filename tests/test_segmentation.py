# pylint: disable=redefined-outer-name,missing-function-docstring

import pytest

import pandas as pd
import SimpleITK as sitk

from pydicer.generate.segmentation import segment_image, segment_dataset, read_segmentation_log
from pydicer.utils import read_converted_data


@pytest.fixture
def test_data_path(tmp_path_factory):
    """Fixture to generate a pydicer style file structure. Along with a few dummy images to
    run a dummy auto-semgentation on."""

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
            1,
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.41813643076347424817314071123",
            "b28321",
            "CT",
            "HNSCC-01-0019",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.233510441938368266923995238123",
            "1.3.6.1.4.1.14519.5.2.1.1706.8040.290727775603409136366833989123",
            "",
            "data/HNSCC-01-0019/images/b28321",
        ],
    ]

    df_converted = pd.DataFrame(rows, columns=cols)
    for _, row in df_converted.iterrows():

        data_obj_path = working_directory.joinpath(row.path)
        data_obj_path.mkdir(parents=True, exist_ok=True)

        img_path = data_obj_path.joinpath("CT.nii.gz")
        sitk.WriteImage(sitk.Image(10, 10, 10, sitk.sitkUInt8), str(img_path))

    converted_path = working_directory.joinpath("data", "HNSCC-01-0019", "converted.csv")
    df_converted.to_csv(converted_path)

    # Also create a dataset directory with converted sub-set
    dataset_path = working_directory.joinpath("test_dataset", "HNSCC-01-0019")
    dataset_path.mkdir(parents=True)
    converted_path = dataset_path.joinpath("converted.csv")
    df_converted[:1].to_csv(converted_path)

    return working_directory


def test_segment_image_incorrect_function_input(test_data_path):
    def seg_func(_, __):
        return {"test": sitk.Image(10, 10, 10, sitk.sitkUInt8)}

    df = read_converted_data(test_data_path)
    img_row = df.iloc[0]

    segment_image(test_data_path, img_row, "test_seg_fail_input", seg_func)

    df_log = read_segmentation_log(image_row=img_row)
    assert len(df_log) == 1
    assert not df_log.iloc[0].success_flag


def test_segment_image_incorrect_function_output(test_data_path):
    def seg_func(img):
        return img

    df = read_converted_data(test_data_path)
    img_row = df.iloc[0]

    segment_image(test_data_path, img_row, "test_seg_fail_output", seg_func)

    df_log = read_segmentation_log(image_row=img_row)
    assert len(df_log) == 1
    assert not df_log.iloc[0].success_flag


def test_segment_image(test_data_path):
    def seg_func(img):
        return {"struct_a": img > 0, "struct_b": img > 1}

    df = read_converted_data(test_data_path)
    img_row = df.iloc[0]

    segment_image(test_data_path, img_row, "test_seg", seg_func)

    df_log = read_segmentation_log(image_row=img_row)
    assert len(df_log) == 1
    assert df_log.iloc[0].success_flag


def test_segment_dataset(test_data_path):
    def seg_func(img):
        return {"struct_a": img > 0, "struct_b": img > 1}

    df = read_converted_data(test_data_path)
    assert len(df) == 2

    segment_dataset(test_data_path, "test_seg", seg_func)

    df = read_converted_data(test_data_path)
    assert len(df) == 4


def test_segment_dataset_subset(test_data_path):
    def seg_func(img):
        return {"struct_a": img > 0, "struct_b": img > 1}

    segment_dataset(test_data_path, "test_seg", seg_func, dataset_name="test_dataset")

    df = read_converted_data(test_data_path, dataset_name="test_dataset")
    assert len(df) == 2
