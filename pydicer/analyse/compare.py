import logging
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from platipy.imaging.label.comparison import compute_volume_metrics, compute_surface_metrics

from pydicer.constants import DEFAULT_MAPPING_ID, CONVERTED_DIR_NAME
from pydicer.dataset.structureset import StructureSet

from pydicer.utils import get_iterator, parse_patient_kwarg, read_converted_data

logger = logging.getLogger(__name__)

AVAILABLE_VOLUME_METRICS = [
    "DSC",
    "volumeOverlap",
    "fractionOverlap",
    "truePositiveFraction",
    "trueNegativeFraction",
    "falsePositiveFraction",
    "falseNegativeFraction",
]

AVAILABLE_SURFACE_METRICS = [
    "hausdorffDistance",
    "meanSurfaceDistance",
    "medianSurfaceDistance",
    "maximumSurfaceDistance",
    "sigmaSurfaceDistance",
    "surfaceDSC",
]


def get_all_similarity_metrics_for_dataset(
    working_directory,
    dataset_name=CONVERTED_DIR_NAME,
    patient=None,
    segment_id=None,
    structure_mapping_id=DEFAULT_MAPPING_ID,
):
    """Return a DataFrame of similarity metrics computed for this dataset.

    Args:
        dataset_name (str, optional): The name of the dataset for which to extract metrics.
            Defaults to CONVERTED_DIR_NAME.
        patient (list|str, optional): A patient ID (or list of patient IDs) to fetch metrics for.
            Defaults to None.
        segment_id (str, optional): Only extract similarity metrics for segment ID. If none is
            supplied then all similarity metrics will be fetched. Defaults to None.
        structure_mapping_id (str, optional): ID of a structure mapping to load computed metrics
            for. Defaults to DEFAULT_MAPPING_ID.

    Returns:
        pd.DataFrame: The DataFrame of all radiomics computed for dataset
    """

    patient = parse_patient_kwarg(patient)

    df_data = read_converted_data(working_directory, dataset_name, patients=patient)

    dfs = []
    for _, struct_row in df_data[df_data["modality"] == "RTSTRUCT"].iterrows():
        struct_dir = Path(struct_row.path)

        if segment_id is None:
            path_glob = struct_dir.glob(f"similarity_*_{structure_mapping_id}.csv")
        else:
            path_glob = struct_dir.glob(f"similarity_{segment_id}_*_{structure_mapping_id}.csv")
        for similarity_file in path_glob:
            col_types = {
                "patient_id": str,
                "hashed_uid_target": str,
                "hashed_uid_reference": str,
                "structure": str,
                "value": float,
            }
            df_metrics = pd.read_csv(similarity_file, index_col=0, dtype=col_types)
            dfs.append(df_metrics)

    if len(dfs) == 0:
        df = pd.DataFrame(
            columns=["patient_id", "hashed_uid_target", "hashed_uid_reference", "structure"]
        )

    df = pd.concat(dfs)
    df.sort_values(
        ["patient_id", "hashed_uid_target", "hashed_uid_reference", "structure"], inplace=True
    )

    return df


def prepare_similarity_metric_analysis(
    working_directory: Union[str, Path],
    analysis_output_directory: Union[str, Path] = None,
    df: pd.DataFrame = None,
    dataset_name: str = CONVERTED_DIR_NAME,
    patient: Union[str, list] = None,
    segment_id: str = None,
    structure_mapping_id: str = DEFAULT_MAPPING_ID,
):
    """Prepare the similarity metric analysis and stores raw metrics and statistics as .csv files
    within the analysis_output_directory. Plots and statistics are also saved as .png files within
    this directory for inspection.

    Args:
        working_directory (Union[str, Path]): The working directory of the PyDicer project.
        analysis_output_directory (Union[str, Path], optional): The directory in which to store the
            output. If none is provided analysis will be generated in a directory named
            similarity_analysis within the working_directory. Defaults to None.
        df (pd.DataFrame, optional): A DataFrame generated using the
            get_all_similarity_metrics_for_dataset function. This might be useful if you wish to
            further filter the DataFrame prior to generating analysis. If none is provided the
            get_all_similarity_metrics_for_dataset will be used to generate the DataFrame. Defaults
            to None.
        dataset_name (str, optional): The name of the dataset to analyse similarity metrics for.
            Defaults to CONVERTED_DIR_NAME.
        patient (Union[str, list], optional): The patients to analyse similarity metrics for.
            Defaults to None.
        segment_id (str, optional): The segment ID to analyse similarity metrics for. Defaults to
            None.
        structure_mapping_id (str, optional): ID of a structure mapping to load computed metrics
            for. Defaults to DEFAULT_MAPPING_ID.
    """

    # Specify a default analysis directory and create it if it doesn't yet exist
    if analysis_output_directory is None:
        analysis_output_directory = working_directory.joinpath("similarity_analysis")

    analysis_output_directory = Path(analysis_output_directory)
    analysis_output_directory.mkdir(exist_ok=True)

    # The user might pass in a dataframe for analysis, if not fetch it.
    if df is None:
        df = get_all_similarity_metrics_for_dataset(
            working_directory=working_directory,
            dataset_name=dataset_name,
            patient=patient,
            segment_id=segment_id,
            structure_mapping_id=structure_mapping_id,
        )

    # Save off the raw metrics to the output folder
    if segment_id is None:
        raw_metrics_output_csv = analysis_output_directory.joinpath(
            f"raw_{structure_mapping_id}.csv"
        )
    else:
        raw_metrics_output_csv = analysis_output_directory.joinpath(
            f"raw_{segment_id}_{structure_mapping_id}.csv"
        )

    df.to_csv(raw_metrics_output_csv)

    # For each metric, generate a plot and a stats csv
    for _, metric in enumerate(df.metric.unique()):
        plt.figure(figsize=(16, 10))
        plt.rcParams.update({"font.size": 22})

        ax = sns.boxplot(data=df[df.metric == metric], x="structure", y="value", hue="segment_id")
        ax.set_ylim([0, max(1, df[df.metric == metric].value.max())])
        ax.set_ylabel(metric)
        ax.set_xlabel("")
        plt.xticks(rotation=45)

        df_stats = (
            df[df.metric == metric][["segment_id", "structure", "value"]]
            .groupby(["segment_id", "structure"])
            .agg(["mean", "std", "max", "min", "count"])
        )
        df_stats = df_stats.reset_index()
        cols = [c[1] if c[1] else c[0] for c in df_stats]
        df_stats.columns = cols
        pd.plotting.table(
            ax=ax,
            data=df_stats.round(2),
            cellLoc="center",
            rowLoc="center",
            loc="bottom",
            bbox=[0.0, -(len(df_stats) * 0.1) - 0.2, 1.0, len(df_stats) * 0.1],
        )

        if segment_id is None:
            stats_output_csv = analysis_output_directory.joinpath(
                f"stats_{metric}_{structure_mapping_id}.csv"
            )
            plot_output = analysis_output_directory.joinpath(
                f"plot_{metric}_{structure_mapping_id}.png"
            )
        else:
            stats_output_csv = analysis_output_directory.joinpath(
                f"stats_{metric}_{segment_id}_{structure_mapping_id}.csv"
            )
            plot_output = analysis_output_directory.joinpath(
                f"plot_{metric}_{segment_id}_{structure_mapping_id}.png"
            )
        df_stats.to_csv(stats_output_csv)

        plt.savefig(plot_output)


def compute_contour_similarity_metrics(
    df_target: pd.DataFrame,
    df_reference: pd.DataFrame,
    segment_id: str,
    mapping_id: str = DEFAULT_MAPPING_ID,
    compute_metrics: list = None,
    force: bool = False,
):
    """_summary_

    Args:
        df_target (pd.DataFrame): DataFrame containing structure set rows to use as target for
            similarity metric computation.
        df_reference (pd.DataFrame): DataFrame containing structure set rows to use as reference
            for similarity metric computation. Each row in reference will be match to target which
            reference the same referenced_sop_instance_uid (image to which they are attached).
        segment_id (str): ID to reference the segmentation for which these metrics are computed.
        mapping_id (str, optional):The mapping ID to use for structure name mapping. Defaults to
            DEFAULT_MAPPING_ID.
        compute_metrics (list, optional): _description_. Defaults to ["DSC", "hausdorffDistance",
            "meanSurfaceDistance", "surfaceDSC"].
        force (bool, optional): If True, metrics will be recomputed even if they have been
            previously computed. Defaults to False.
    """

    # Merge the DataFrames to have a row for each target-reference combination based on the image
    # they are referencing
    df = pd.merge(
        df_target,
        df_reference,
        on="referenced_sop_instance_uid",
        suffixes=("_target", "_reference"),
    )

    if compute_metrics is None:
        compute_metrics = ["DSC", "hausdorffDistance", "meanSurfaceDistance", "surfaceDSC"]

    # For each pair of structures, compute similarity metrics
    for _, row in get_iterator(
        df.iterrows(), length=len(df), unit="structure sets", name="Compare Structures"
    ):
        target_path = Path(row.path_target)
        similarity_csv = target_path.joinpath(
            f"similarity_{segment_id}_{row.hashed_uid_reference}_{mapping_id}.csv"
        )
        if similarity_csv.exists() and not force:
            logger.info("Similarity metrics already computed at %s", similarity_csv)
            continue

        results = []

        ss_target = StructureSet(
            df_target[df_target.hashed_uid == row.hashed_uid_target].iloc[0], mapping_id=mapping_id
        )
        ss_reference = StructureSet(
            df_reference[df_reference.hashed_uid == row.hashed_uid_reference].iloc[0],
            mapping_id=mapping_id,
        )

        for structure, mask_target in ss_target.items():
            if structure in ss_reference.get_unmapped_structures():
                for metric in compute_metrics:
                    result_entry = {
                        "patient_id": row.patient_id_target,
                        "hashed_uid_target": row.hashed_uid_target,
                        "hashed_uid_reference": row.hashed_uid_reference,
                        "structure": structure,
                        "metric": metric,
                        "value": np.nan,
                    }
                    results.append(result_entry)

                logger.warning(
                    "No reference structure found for %s in %s. Available structure names are: %s",
                    structure,
                    row.hashed_uid_reference,
                    ss_reference.unmapped_structure_names,
                )

                continue

            mask_reference = ss_reference[structure]

            volume_metrics = {}
            if set(compute_metrics).intersection(set(AVAILABLE_VOLUME_METRICS)):
                volume_metrics = compute_volume_metrics(mask_target, mask_reference)

            surface_metrics = {}
            if set(compute_metrics).intersection(set(AVAILABLE_SURFACE_METRICS)):
                surface_metrics = compute_surface_metrics(mask_target, mask_reference)

            metrics = {**volume_metrics, **surface_metrics}

            for metric in compute_metrics:
                result_entry = {
                    "patient_id": row.patient_id_target,
                    "hashed_uid_target": row.hashed_uid_target,
                    "hashed_uid_reference": row.hashed_uid_reference,
                    "segment_id": segment_id,
                    "structure": structure,
                    "metric": metric,
                    "value": metrics[metric],
                }
                results.append(result_entry)

        df_results = pd.DataFrame(results)
        df_results.to_csv(similarity_csv)
