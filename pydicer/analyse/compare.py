import pandas as pd

from platipy.imaging.label.comparison import compute_volume_metrics, compute_surface_metrics

from pydicer.constants import DEFAULT_MAPPING_ID
from pydicer.dataset.structureset import StructureSet


def compute_contour_similarity_metrics(df_target, df_reference, mapping_id=DEFAULT_MAPPING_ID):
    # Merge the DataFrames to have a row for each target-reference combination based on the image
    # they are referencing
    df = pd.merge(
        df_target,
        df_reference,
        on="referenced_sop_instance_uid",
        suffixes=("_target", "_reference"),
    )

    # For each pair of structures, compute similarity metrics
    for idx, row in df.iterrows():
        ss_target = StructureSet(
            df[df.hashed_uid == row.hashed_uid_target].iloc[0], mapping_id=mapping_id
        )
        ss_reference = StructureSet(
            df[df.hashed_uid == row.hashed_uid_target].iloc[0], mapping_id=mapping_id
        )
