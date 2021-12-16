import os
import json
import logging
from datetime import datetime
from pathlib import Path

import pydicom

logger = logging.getLogger(__name__)


def determine_dcm_datetime(ds):

    if "SeriesDate" in ds and len(ds.SeriesDate) > 0:

        if "SeriesTime" in ds and len(ds.SeriesTime) > 0:
            date_time_str = f"{ds.SeriesDate}{ds.SeriesTime}"
            if "." in date_time_str:
                return datetime.strptime(date_time_str, "%Y%m%d%H%M%S.%f")
            else:
                return datetime.strptime(date_time_str, "%Y%m%d%H%M%S")

        return datetime.strptime(ds.SeriesDate, "%Y%m%d")

    if "StudyDate" in ds and len(ds.StudyDate) > 0:

        if "StudyTime" in ds and len(ds.StudyTime) > 0:
            date_time_str = f"{ds.StudyDate}{ds.StudyTime}"
            if "." in date_time_str:
                return datetime.strptime(date_time_str, "%Y%m%d%H%M%S.%f")
            else:
                return datetime.strptime(date_time_str, "%Y%m%d%H%M%S")

        return datetime.strptime(ds.StudyDate, "%Y%m%d")

    if "InstanceCreationDate" in ds and len(ds.InstanceCreationDate) > 0:

        if "InstanceCreationTime" in ds and len(ds.InstanceCreationTime) > 0:
            date_time_str = f"{ds.InstanceCreationDate}{ds.InstanceCreationTime}"
            if "." in date_time_str:
                return datetime.strptime(date_time_str, "%Y%m%d%H%M%S.%f")
            else:
                return datetime.strptime(date_time_str, "%Y%m%d%H%M%S")

        return datetime.strptime(ds.InstanceCreationDate, "%Y%m%d")

    return None


def rt_latest_struct(working_directory, dataset_name, patients=None, **kwargs):
    """Select the latest Structure set and the image which it is linked to. You can specify keyword
    arguments to for a match on any top level DICOM attributes. You may also supply lists of values
    to these, one of which should match to select that series.

    Example of matching the latest structure set with Series Description being "FINAL" or
    "APPROVED"
    .. code-block:: python

        prepare_dataset = PrepareDataset(output_directory)
        prepare_dataset.prepare(
            "./clean",
            "rt_latest_struct",
            SeriesDescription=["FINAL", "APPROVED"]
        )

    Args:
        working_directory (pathlib.Path): The directory holding the converted data
        dataset_name (str): The name of the dataset being prepared
    """

    data_directory = working_directory.joinpath("data")
    target_directory = working_directory.joinpath(dataset_name)
    pat_dirs = [p for p in data_directory.glob("*") if p.is_dir() and not "quarantine" in p.name]

    if len(pat_dirs) == 0:
        logger.warning("No patient directories found in directory")

    for pat_dir in pat_dirs:

        pat_id = pat_dir.name

        if patients:
            if not pat_id in patients:
                continue

        logger.debug("Selecting data for patient: %s", pat_id)

        structure_metadata = list(pat_dir.glob("**/structures/*.json"))

        pat_structs = []
        for struct_md in structure_metadata:

            with open(struct_md, "r", encoding="utf8") as json_file:
                ds_dict = json.load(json_file)

            struct_ds = pydicom.Dataset.from_json(ds_dict, bulk_data_uri_handler=lambda _: None)
            ds_date = determine_dcm_datetime(struct_ds)

            skip_series = False
            for k in kwargs:

                if not k in struct_ds:
                    logger.debug("Attribute %s not in %s", k, struct_md)
                    skip_series = True
                    continue

                if isinstance(kwargs[k], str):
                    kwargs[k] = [kwargs[k]]

                attribute_match = False
                for sd in kwargs[k]:
                    if sd == struct_ds[k].value:
                        attribute_match = True

                if not attribute_match:
                    skip_series = True
                    logger.debug(
                        "Attribute %s's value(s) %s does not match %s",
                        k,
                        kwargs[k],
                        str(struct_ds[k].value),
                    )

            if skip_series:
                logger.debug("Skipping series based on filters: %s", struct_md)
                continue

            pat_structs.append({"metadata": struct_md, "datetime": ds_date})

            logger.debug("Found structure metadata: %s with date: %s", struct_md, ds_date)

        pat_structs = sorted(pat_structs, key=lambda d: d["datetime"], reverse=True)

        if len(pat_structs) == 0:
            logger.warning("No data selected for patient: %s", pat_id)
            continue

        # Select the latest structure
        pat_struct = pat_structs[0]
        logger.debug(
            "Selecting structure metadata: %s with date: %s",
            pat_struct["metadata"],
            pat_struct["datetime"],
        )

        struct_md = pat_struct["metadata"]
        struct_dir_name = struct_md.name.replace(struct_md.suffix, "")
        struct_parts = struct_dir_name.split("_")
        referenced_img_id = struct_parts[1]

        pat_prep_dir = target_directory.joinpath(pat_id)

        files_to_link = list(pat_dir.glob(f"**/images/*{referenced_img_id}*")) + list(
            pat_dir.glob(f"**/structures/*{struct_dir_name}*")
        )

        for file_path in files_to_link:
            symlink_path = pat_prep_dir.joinpath(file_path.parent.name, file_path.name)
            if symlink_path.exists():
                os.remove(symlink_path)

            rel_part = os.sep.join(
                [".." for _ in symlink_path.parent.relative_to(working_directory).parts]
            )
            src_path = Path(f"{rel_part}{os.sep}{file_path.relative_to(working_directory)}")

            symlink_path.parent.mkdir(parents=True, exist_ok=True)
            symlink_path.symlink_to(src_path)