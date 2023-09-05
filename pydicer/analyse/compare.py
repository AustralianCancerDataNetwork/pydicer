import logging
from pathlib import Path

import pandas as pd
import numpy as np

from platipy.imaging.label.comparison import compute_volume_metrics, compute_surface_metrics

from pydicer.constants import DEFAULT_MAPPING_ID
from pydicer.dataset.structureset import StructureSet

from pydicer.utils import get_iterator

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


def compute_contour_similarity_metrics(
    df_target: pd.DataFrame,
    df_reference: pd.DataFrame,
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
        similarity_csv = target_path.joinpath(f"similarity_{row.hashed_uid_reference}.csv")
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
                    "structure": structure,
                    "metric": metric,
                    "value": metrics[metric],
                }
                results.append(result_entry)

        df_results = pd.DataFrame(results)
        df_results.to_csv(similarity_csv)
