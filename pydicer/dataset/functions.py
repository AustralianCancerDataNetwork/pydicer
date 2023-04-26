import logging

import pandas as pd

from pydicer.utils import load_object_metadata, determine_dcm_datetime

logger = logging.getLogger(__name__)


def rt_latest_struct(df, **kwargs):
    """Select the latest Structure set and the image which it is linked to. You can specify keyword
    arguments to for a match on any top level DICOM attributes. You may also supply lists of values
    to these, one of which should match to select that series.

    Example of matching the latest structure set with Series Description being "FINAL" or
    "APPROVED"

    .. code-block:: python

        prepare_dataset = PrepareDataset(working_directory)
        prepare_dataset.prepare(
            "clean",
            "rt_latest_struct",
            SeriesDescription=["FINAL", "APPROVED"]
        )

    Args:
        df (pd.DataFrame): DataFrame of converted data objects available for dataset

    Returns:
        pd.DataFrame: The filtered DataFrame containing only the objects to select
    """

    keep_rows = []

    for pat_id, df_patient in df.groupby("patient_id"):

        logger.debug("Selecting data for patient: %s", pat_id)

        df_patient["datetime"] = pd.NaT

        struct_indicies = []
        for idx, row in df_patient[df_patient["modality"] == "RTSTRUCT"].iterrows():

            struct_ds = load_object_metadata(row)
            ds_date = determine_dcm_datetime(struct_ds)
            df_patient.loc[idx, "datetime"] = ds_date

            skip_series = False
            for k, item in kwargs.items():

                if not k in struct_ds:
                    logger.debug("Attribute %s not in metadata", k)
                    skip_series = True
                    continue

                if isinstance(item, str):
                    item = [item]

                attribute_match = False
                for sd in item:
                    if sd == struct_ds[k].value:
                        attribute_match = True

                if not attribute_match:
                    skip_series = True
                    logger.debug(
                        "Attribute %s's value(s) %s does not match %s",
                        k,
                        item,
                        str(struct_ds[k].value),
                    )

            if skip_series:
                logger.debug("Skipping series based on filters")
                continue

            struct_indicies.append(idx)

        df_structures = df_patient.loc[struct_indicies]
        df_structures.sort_values("datetime", ascending=False, inplace=True)

        if len(df_structures) == 0:
            logger.warning("No data selected for patient: %s", pat_id)
            continue

        # Select the latest structure
        struct_row = df_structures.iloc[0]
        logger.debug(
            "Selecting structure: %s with date: %s",
            struct_row["hashed_uid"],
            struct_row["datetime"],
        )
        keep_rows.append(struct_row.name)  # Track index of row to keep

        # Find the linked image
        df_linked_img = df[df["sop_instance_uid"] == struct_row.referenced_sop_instance_uid]

        if len(df_linked_img) == 0:
            logger.warning("No linked images found for structure: %s", struct_row.hashed_uid)
            continue

        keep_rows.append(df_linked_img.iloc[0].name)  # Keep the index of the row of the image too

    return df.loc[keep_rows]


def rt_latest_dose(df, **kwargs):
    """Select the latest RTDOSE and the image, structure and plan which it is linked to. You can
    specify keyword arguments to for a match on any top level DICOM attributes. You may also supply
    lists of values to these, one of which should match to select that series.

    Example of matching the latest dose with Series Description being "FINAL" or "APPROVED"

    .. code-block:: python

        prepare_dataset = PrepareDataset(working_directory)
        prepare_dataset.prepare(
            "clean",
            "rt_latest_dose",
            SeriesDescription=["FINAL", "APPROVED"]
        )

    Args:
        df (pd.DataFrame): DataFrame of converted data objects available for dataset

    Returns:
        pd.DataFrame: The filtered DataFrame containing only the objects to select
    """

    patients = df.patient_id.unique()

    keep_rows = []

    for pat_id in patients:

        logger.debug("Selecting data for patient: %s", pat_id)

        df_patient = df[df["patient_id"] == pat_id]

        df_doses = df_patient[df_patient["modality"] == "RTDOSE"]

        dose_indicies = []
        dose_dates = []
        for idx, row in df_doses.iterrows():

            dose_ds = load_object_metadata(row)
            ds_date = determine_dcm_datetime(dose_ds)
            dose_dates.append(ds_date)

            skip_series = False
            for k, item in kwargs.items():

                if not k in dose_ds:
                    logger.debug("Attribute %s not in metadata", k)
                    skip_series = True
                    continue

                if isinstance(item, str):
                    item = [item]

                attribute_match = False
                for sd in item:
                    if sd == dose_ds[k].value:
                        attribute_match = True

                if not attribute_match:
                    skip_series = True
                    logger.debug(
                        "Attribute %s's value(s) %s does not match %s",
                        k,
                        item,
                        str(dose_ds[k].value),
                    )

            if skip_series:
                logger.debug("Skipping series based on filters")
                continue

            dose_indicies.append(idx)

        df_doses = df_doses.assign(datetime=dose_dates)
        df_doses = df_doses.loc[dose_indicies]
        df_doses.sort_values("datetime", ascending=False, inplace=True)

        if len(df_doses) == 0:
            logger.warning("No data selected for patient: %s", pat_id)
            continue

        # Select the latest structure
        dose_row = df_doses.iloc[0]
        logger.debug(
            "Selecting dose: %s with date: %s",
            dose_row["hashed_uid"],
            dose_row["datetime"],
        )
        keep_rows.append(dose_row.name)  # Track index of row of dose to keep

        # Find the linked plan
        df_linked_plan = df[df["sop_instance_uid"] == dose_row.referenced_sop_instance_uid]

        if len(df_linked_plan) == 0:
            logger.warning("No linked plans found for dose: %s", dose_row.sop_instance_uid)
            continue

        # Find the linked structure set
        plan_row = df_linked_plan.iloc[0]
        keep_rows.append(plan_row.name)  # Keep the index of the row of the plan
        df_linked_struct = df[df["sop_instance_uid"] == plan_row.referenced_sop_instance_uid]

        if len(df_linked_struct) == 0:
            # Try to link via Frame of Reference instead
            df_linked_struct = df[
                (df["modality"] == "RTSTRUCT") & (df["for_uid"] == dose_row.for_uid)
            ]

        if len(df_linked_struct) == 0:
            logger.warning("No structures found for plan: %s", plan_row.sop_instance_uid)
            continue

        # Find the linked image
        struct_row = df_linked_struct.iloc[0]
        keep_rows.append(struct_row.name)  # Keep the index of the row of the structure
        df_linked_img = df[df["sop_instance_uid"] == struct_row.referenced_sop_instance_uid]

        if len(df_linked_img) == 0:
            logger.warning("No linked images found for structure: %s", struct_row.hashed_uid)
            continue

        keep_rows.append(df_linked_img.iloc[0].name)  # Keep the index of the row of the image too

    return df.loc[keep_rows]
