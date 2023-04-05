import json
import logging
import os
from pathlib import Path

import pandas as pd

from pydicer.constants import CONVERTED_DIR_NAME

from pydicer.dataset import functions
from pydicer.utils import read_converted_data, map_contour_name

logger = logging.getLogger(__name__)


class PrepareDataset:
    def __init__(self, working_directory="."):
        self.working_directory = Path(working_directory)

    def add_object_to_dataset(self, dataset_name, data_object_row):
        """Add one data object to a dataset.

        Args:
            dataset_name (str): The name of the dataset to add the object to.
            data_object_row (pd.Series): The DataFrame row of the converted object.
        """

        dataset_dir = self.working_directory.joinpath(dataset_name)

        # Create a copy so that we aren't manuipulating the original entry
        data_object_row = data_object_row.copy()

        if data_object_row.path.startswith(str(self.working_directory)):
            data_object_row.path = str(
                Path(data_object_row.path).relative_to(self.working_directory)
            )

        object_path = Path(data_object_row.path)
        symlink_path = dataset_dir.joinpath(object_path.relative_to(CONVERTED_DIR_NAME))

        rel_part = os.sep.join(
            [".." for _ in symlink_path.parent.relative_to(self.working_directory).parts]
        )
        src_path = Path(f"{rel_part}{os.sep}{object_path}")

        symlink_path.parent.mkdir(parents=True, exist_ok=True)

        if symlink_path.exists():
            logger.debug("Symlink path already exists: %s", symlink_path)
        else:
            symlink_path.symlink_to(src_path)

        pat_id = data_object_row.patient_id
        pat_dir = dataset_dir.joinpath(pat_id)
        pat_converted_csv = pat_dir.joinpath("converted.csv")
        df_pat = pd.DataFrame([data_object_row])
        if pat_converted_csv.exists():

            col_types = {"patient_id": str, "hashed_uid": str}
            df_converted = pd.read_csv(pat_converted_csv, index_col=0, dtype=col_types)

            # Check if this object already exists in the converted dataframe
            if len(df_converted[df_converted.hashed_uid == data_object_row.hashed_uid]) == 0:
                # If not add it
                df_pat = pd.concat([df_converted, df_pat])
            else:
                # Otherwise just leave the converted data as is
                df_pat = df_converted

        df_pat = df_pat.reset_index(drop=True)
        df_pat.to_csv(pat_dir.joinpath("converted.csv"))

    def prepare_from_dataframe(self, dataset_name, df_prepare):
        """Prepare a dataset from a filtered converted dataframe

        Args:
            dataset_name (str): The name of the dataset to generate
            df_prepare (pd.DataFrame): Filtered Pandas DataFrame containing rows of converted data.
        """

        dataset_dir = self.working_directory.joinpath(dataset_name)
        if dataset_dir.exists():
            logger.warning(
                "Dataset directory already exists. Consider using a different dataset name or "
                "remove the existing directory"
            )

        # Remove the working directory part for when we re-save off the filtered converted csv
        df_prepare.path = df_prepare.path.apply(
            lambda p: str(Path(p).relative_to(self.working_directory))
        )

        # For each data object prepare the data in the dataset directory
        for _, row in df_prepare.iterrows():
            self.add_object_to_dataset(dataset_name, row)

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

        if isinstance(preparation_function, str):

            preparation_function = getattr(functions, preparation_function)

        if not callable(preparation_function):
            raise AttributeError(
                "preparation_function must be a function or a str defined in pydicer.dataset"
            )

        logger.info("Preparing dataset %s using function: %s", dataset_name, preparation_function)

        # Grab the DataFrame containing all the converted data
        df_converted = read_converted_data(self.working_directory, patients=patients)

        # Send to the prepare function which will return a DataFrame of the data objects to use for
        # the dataset
        df_clean_data = preparation_function(df_converted, **kwargs)

        self.prepare_from_dataframe(dataset_name, df_clean_data)


class MapStructureSetNomenclature:
    """Class to handle the mapping of structure set nomenclature"""

    def __init__(self, working_directory: Path):
        self.working_directory = Path(working_directory)
        self.project_structs_map_path = self.working_directory.joinpath(".pydicer").joinpath(
            "structures_map.json"
        )

    def map_project_structure_set_names(
        self,
    ):
        """Function to perform the mapping of all structures in the converted "data" folder using
        a project-wide mapping file
        """
        try:
            with open(self.project_structs_map_path, "r", encoding="utf8") as structs_map_file:
                struct_map_dict = json.load(structs_map_file)["structures"]
                pat_ids = read_converted_data(self.working_directory).patient_id.unique()
                # Get all patients
                for pat_id in pat_ids:
                    pat_struct_sets_path = (
                        self.working_directory.joinpath("data")
                        .joinpath(pat_id)
                        .joinpath("structures")
                    )
                    # Get all structure sets for this patient
                    for p in pat_struct_sets_path.rglob("*"):
                        if p.is_dir():
                            df = pd.DataFrame(columns=["old_structure_name", "path_to_structure"])
                            # Grab the names of the structures for this set, as well as the paths
                            # to these NifTi files
                            df.old_structure_name, df.path_to_structure = (
                                [
                                    str(x.name.strip(".nii.gz"))
                                    for x in p.glob("*nii.gz")
                                    if x.is_file()
                                ],
                                [str(x) for x in p.glob("*nii.gz") if x.is_file()],
                            )
                            logger.debug("Mapping names for structure set: %s", p.name)
                            df.apply(
                                lambda x: map_contour_name(
                                    x.old_structure_name,
                                    x.path_to_structure,
                                    struct_map_dict,
                                    "Project",
                                ),
                                axis=1,
                            )
        except FileNotFoundError:
            logger.error(
                """'%s' structures mapping file not
                found for the project!""",
                self.project_structs_map_path,
            )

    def map_specific_structure_set_names(
        self, struct_set_id, mapping_file_name="structures_map.json"
    ):
        """Function to map a specific structure set structures' names according to its own mapping
        file.

        Args:
            struct_set_id (str): the hashed id of the structure set to be mapped
            mapping_file_name (str, optional): name of the mapping file that must sit under the
            "struct_set_id" directory. Defaults to "structures_map.json".
        """
        try:
            df_converted = read_converted_data(self.working_directory)
            patient_structs_map_path = Path(
                df_converted[df_converted.hashed_uid == struct_set_id].path.iloc[0]
            )
            with open(
                patient_structs_map_path.joinpath(mapping_file_name), "r", encoding="utf8"
            ) as structs_map_file:
                struct_map_dict = json.load(structs_map_file)["structures"]
                if patient_structs_map_path.is_dir():
                    df = pd.DataFrame(columns=["old_structure_name", "path_to_structure"])
                    # Grab the names of the structures for this set, as well as the paths
                    # to these NifTi files
                    df.old_structure_name, df.path_to_structure = (
                        [
                            str(x.name.strip(".nii.gz"))
                            for x in patient_structs_map_path.glob("*nii.gz")
                            if x.is_file()
                        ],
                        [str(x) for x in patient_structs_map_path.glob("*nii.gz") if x.is_file()],
                    )
                    logger.debug(
                        "Mapping names for structure set: %s", patient_structs_map_path.name
                    )
                    df.apply(
                        lambda x: map_contour_name(
                            x.old_structure_name, x.path_to_structure, struct_map_dict, "Patient"
                        ),
                        axis=1,
                    )
        except FileNotFoundError:
            logger.error(
                """ '%s' structures mapping file not
                found for patient ({pat_id}) structure set ({struct_set_id})!""",
                mapping_file_name,
            )
