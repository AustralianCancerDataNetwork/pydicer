import os
import logging
from pathlib import Path

import pandas as pd

from pydicer.dataset import functions

logger = logging.getLogger(__name__)


def read_converted_data_frame(data_directory, patients=None):
    """Read the converted data frame from the supplied data directory.

    Args:
        data_directory (str): The directory in which data was stored.
        patients (list, optional): The list of patients for which to read converted data. If None
            is supplied then all data will be read. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with the converted data objects.
    """

    df_result = None

    for pat_dir in data_directory.glob("*"):

        if not pat_dir.is_dir():
            continue

        pat_id = pat_dir.name

        if patients is not None:
            if pat_id not in patients:
                continue

        # Read in the DataFrame storing the converted data for this patient
        converted_csv = data_directory.joinpath(pat_id, "converted.csv")
        if not converted_csv.exists():
            logger.warning("Converted CSV doesn't exist for %s", pat_id)
            continue

        df_converted = pd.read_csv(converted_csv, index_col=0)

        if df_result is None:
            df_result = df_converted
        else:
            df_result = df_result.append(df_converted)

    return df_result.reset_index(drop=True)


class PrepareDataset:
    def __init__(self, working_directory="."):
        self.working_directory = Path(working_directory)

    def prepare(self, dataset_name, preparation_function, patients=None, **kwargs):
        """Calls upon an appropriate preparation function to generate a clean dataset ready for
        use. Additional keyword arguments are passed through to the preparation_function.

        Args:
            dataset_name (str): The name of the dataset to generate
            preparation_function (function|str): the function use for preparation
            patients (list): The list of patient IDs to use for dataset. If None then all patients
                will be considered. Defaults to None.

        Raises:
            AttributeError: Raised if preparation_function is not a function or a string defining
              a known preparation function.
        """

        dataset_dir = self.working_directory.joinpath(dataset_name)
        if dataset_dir.exists():
            logger.warning(
                "Dataset directory already exists. Consider using a different dataset name or "
                "remove the existing directory"
            )

        if isinstance(preparation_function, str):

            preparation_function = getattr(functions, preparation_function)

        if not callable(preparation_function):
            raise AttributeError(
                "preparation_function must be a function or a str defined in pydicer.dataset"
            )

        logger.info("Preparing dataset %s using function: %s", dataset_name, preparation_function)

        # Grab the DataFrame containing all the converted data
        converted_path = self.working_directory.joinpath("data")
        df_converted = read_converted_data_frame(converted_path, patients=patients)

        # Send to the prepare function which will return a DataFrame of the data objects to use for
        # the dataset
        df_clean_data = preparation_function(df_converted, **kwargs)

        # For each data object prepare the data in the clean directory (using symbolic links)
        for _, row in df_clean_data.iterrows():

            object_path = Path(row.path)

            symlink_path = dataset_dir.joinpath(object_path.relative_to(converted_path))

            rel_part = os.sep.join(
                [".." for _ in symlink_path.parent.relative_to(self.working_directory).parts]
            )
            src_path = Path(f"{rel_part}{os.sep}{object_path.relative_to(self.working_directory)}")

            symlink_path.parent.mkdir(parents=True, exist_ok=True)

            if symlink_path.exists():
                logger.debug("Symlink path already exists: %s", symlink_path)
            else:
                symlink_path.symlink_to(src_path)

        # Save off the converted data for each patient
        for pat_id, df in df_clean_data.groupby("patient_id"):

            pat_dir = dataset_dir.joinpath(pat_id)

            df.to_csv(pat_dir.joinpath("converted.csv"))
