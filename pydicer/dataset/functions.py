import os
import json
import logging
from datetime import datetime

import pydicom

logger = logging.getLogger(__name__)


def determine_dcm_datetime(ds):

    if "SeriesDate" in ds and len(ds.SeriesDate) > 0:

        if "SeriesTime" in ds and len(ds.StudyTime) > 0:
            return datetime.strptime(f"{ds.SeriesDate}{ds.StudyTime}", "%Y%m%d%H%M%S")

        return datetime.strptime(ds.SeriesDate, "%Y%m%d")

    if "StudyDate" in ds and len(ds.StudyDate) > 0:

        if "StudyTime" in ds and len(ds.StudyTime) > 0:
            return datetime.strptime(f"{ds.StudyDate}{ds.StudyTime}", "%Y%m%d%H%M%S")

        return datetime.strptime(ds.StudyDate, "%Y%m%d")

    if "InstanceCreationDate" in ds and len(ds.InstanceCreationDate) > 0:

        if "InstanceCreationTime" in ds and len(ds.InstanceCreationTime) > 0:
            return datetime.strptime(
                f"{ds.InstanceCreationDate}{ds.InstanceCreationTime}", "%Y%m%d%H%M%S"
            )

        return datetime.strptime(ds.InstanceCreationDate, "%Y%m%d")

    return None


def rt_latest_struct(data_directory, target_directory, **kwargs):
    """Select the latest Structure set and the image which it is linked to. You can specify keyword
    arguments to for a match on any top level DICOM attributes. You may also supply lists of values
    to these, one of which should match to select that series.

    Example of matching the latest structure set with Series Description being "FINAL" or
    "APPROVED"
    .. code-block:: python

    prepare_dataset = PrepareDataset(output_directory)
    prepare_dataset.prepare("./clean", "rt_latest_struct", SeriesDescription=["FINAL", "APPROVED"])

    Args:
        data_directory (pathlib.Path): The directory holding the converted data
        target_directory (pathlib.Path): The directory in which to place the linked clean dataset
    """

    pat_dirs = [p for p in data_directory.glob("*") if p.is_dir() and not "quarantine" in p.name]

    if len(pat_dirs) == 0:
        logger.warning("No patient directories found in directory")

    for pat_dir in pat_dirs:

        pat_id = pat_dir.name

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
        pat_prep_img_dir = pat_prep_dir.joinpath("images")
        pat_prep_struct_dir = pat_prep_dir.joinpath("structure")
        pat_prep_img_dir.mkdir(parents=True, exist_ok=True)
        pat_prep_struct_dir.mkdir(parents=True, exist_ok=True)

        files_to_link = list(pat_dir.glob(f"**/images/*{referenced_img_id}*")) + list(
            pat_dir.glob(f"**/structures/*{struct_dir_name}*")
        )

        for img_file in files_to_link:
            symlink_path = pat_prep_img_dir.joinpath(img_file.name)
            if symlink_path.exists():
                os.remove(symlink_path)
            symlink_path.symlink_to(img_file.resolve())
