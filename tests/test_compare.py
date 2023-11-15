# pylint: disable=redefined-outer-name,missing-function-docstring

import tempfile

from pathlib import Path
import pandas as pd

from pydicer import PyDicer
from pydicer.analyse.compare import (
    compute_contour_similarity_metrics,
    get_all_similarity_metrics_for_dataset,
    prepare_similarity_metric_analysis,
)
from pydicer.utils import read_converted_data


def test_compare_auto_segmentations(test_data_autoseg):
    working_directory = test_data_autoseg
    df = read_converted_data(working_directory=working_directory)

    # We'll test this by comparing the structures against themselves,
    # hence we expect perfect metrics
    df_target = df[df.modality == "RTSTRUCT"]
    df_reference = df[df.modality == "RTSTRUCT"]

    PyDicer(working_directory)
    segment_id = "test_seg"
    compute_contour_similarity_metrics(df_target, df_reference, segment_id)

    df_stats = get_all_similarity_metrics_for_dataset(working_directory)

    assert len(df_stats) == 200

    df_dsc = df_stats[df_stats["metric"] == "DSC"]
    assert df_dsc.value.min() == 1.0
    assert df_dsc.value.max() == 1.0


def test_compaare_metrics_analysis(test_data_autoseg):
    working_directory = test_data_autoseg
    df = read_converted_data(working_directory=working_directory)

    # We'll test this by comparing the structures against themselves,
    # hence we expect perfect metrics
    df_target = df[df.modality == "RTSTRUCT"]
    df_reference = df[df.modality == "RTSTRUCT"]

    PyDicer(working_directory)
    segment_id = "test_seg"
    compute_contour_similarity_metrics(df_target, df_reference, segment_id)

    with tempfile.TemporaryDirectory() as analysis_dir:
        analysis_dir = Path(analysis_dir)

        prepare_similarity_metric_analysis(
            working_directory=working_directory,
            analysis_output_directory=analysis_dir,
            segment_id=segment_id,
        )

        # Check that the output files exist
        raw_metrics_file = analysis_dir.joinpath("raw_test_seg_default.csv")
        assert raw_metrics_file.exists()
        stats_metrics_file = analysis_dir.joinpath("stats_test_seg_default.csv")
        assert stats_metrics_file.exists()
        plot_dsc_file = analysis_dir.joinpath("plot_DSC_test_seg_default.png")
        assert plot_dsc_file.exists()
        plot_hd_file = analysis_dir.joinpath("plot_hausdorffDistance_test_seg_default.png")
        assert plot_hd_file.exists()
        plot_msd_file = analysis_dir.joinpath("plot_meanSurfaceDistance_test_seg_default.png")
        assert plot_msd_file.exists()
        plot_sdsc_file = analysis_dir.joinpath("plot_surfaceDSC_test_seg_default.png")
        assert plot_sdsc_file.exists()

        # Read in the raw metrics file and do some checks
        df_raw = pd.read_csv(raw_metrics_file, index_col=0)
        assert len(df_raw) == 200

        # Since these structures compared against themselves, expect perfect metrics
        assert df_raw[df_raw.metric == "DSC"].value.min() == 1.0
        assert df_raw[df_raw.metric == "surfaceDSC"].value.min() == 1.0
        assert df_raw[df_raw.metric == "hausdorffDistance"].value.max() == 0.0
        assert df_raw[df_raw.metric == "meanSurfaceDistance"].value.max() == 0.0

        # Read in the stats metrics file and do some checks
        df_stats = pd.read_csv(stats_metrics_file, index_col=0)
        assert len(df_stats) == 36

        # Check one fo the rows
        row_check = df_stats[
            (df_stats.structure == "Esophagus") & (df_stats.metric == "surfaceDSC")
        ].iloc[0]
        assert row_check["mean"] == 1.0
        assert row_check["std"] == 0.0
        assert row_check["max"] == 1.0
        assert row_check["min"] == 1.0
        assert row_check["count"] == 10
