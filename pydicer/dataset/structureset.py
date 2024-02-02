import json
import logging
from pathlib import Path

import SimpleITK as sitk
import pandas as pd

from pydicer.constants import DEFAULT_MAPPING_ID

logger = logging.getLogger(__name__)


def get_mapping_for_structure_set(
    structure_set_row: pd.Series, mapping_id: str
) -> dict:
    """Searches the folder hierarchy to find a structure name mapping file with the given ID.

    Args:
        structure_set_row (pd.Series): The converted dataframe row entry for the structure set.
        mapping_id (str): The ID of the mapping to find.

    Returns:
        dict: The structure name mapping
    """
    structure_set_path = Path(structure_set_row.path)

    potential_mapping_paths = [
        # First look in the structure_set_path folder for the structure mapping
        structure_set_path.joinpath(".structure_set_mappings"),
        # Next look in the patient folder
        structure_set_path.parent.joinpath(".structure_set_mappings"),
        # Finally look for the project wide mapping
        structure_set_path.parent.parent.parent.parent.joinpath(
            ".pydicer", ".structure_set_mappings"
        ),
    ]

    for mapping_path in potential_mapping_paths:
        mapping_file = mapping_path.joinpath(f"{mapping_id}.json")
        if mapping_file.exists():
            logger.debug("Using mapping file in %s", mapping_file)
            with open(mapping_file, encoding="utf-8") as json_file:
                return json.load(json_file)

    return None


class StructureSet(dict):
    def __init__(self, structure_set_row, mapping_id=DEFAULT_MAPPING_ID):
        if not structure_set_row.modality == "RTSTRUCT":
            raise AttributeError("structure_set_row modality must be RTSTRUCT")

        self.structure_set_path = Path(structure_set_row.path)
        self.structure_set_id = structure_set_row.hashed_uid

        self.structure_names = [
            s.name.replace(".nii.gz", "")
            for s in self.structure_set_path.glob("*.nii.gz")
        ]
        self.unmapped_structure_names = self.structure_names

        self.structure_mapping = None

        # Check if we can find a mapping for this structure set, if not we'll just used the
        # unmapped structure names
        if mapping_id is not None:
            self.structure_mapping = get_mapping_for_structure_set(
                structure_set_row, mapping_id
            )

            if self.structure_mapping is None:
                logger.warning("No mapping file found with id %s", mapping_id)

        if self.structure_mapping is not None:
            self.structure_names = list(self.structure_mapping.keys())

        self.cache = {}

    def get_mapped_structure_name(self, item: str) -> str:
        """Get the structure set specific name for a structure that may have been mapped.

        Args:
            item (str): The standardised name to look up.

        Returns:
            str: The structure set specific name if it could be mapped (returns the original name
              otherwise).
        """
        structure_name = item

        if self.structure_mapping is not None:
            if item in self.structure_mapping:
                for variation in self.structure_mapping[item]:
                    variation_path = self.structure_set_path.joinpath(
                        f"{variation}.nii.gz"
                    )
                    if variation_path.exists():
                        # Found variation, let's use that file...
                        # TODO an issue would occur if there were multiple files that would match
                        # this mapping. In that case we should probably throw an error (or at
                        # a warning?).
                        structure_name = variation

        return structure_name

    def get_standardised_structure_name(self, item: str) -> str:
        """Get the standardised name for a structure that is present in this structure set.

        Args:
            item (str): The name of the structure in this structure set.

        Returns:
            str: The standardised name if it could be mapped (returns the original name
              otherwise).
        """

        structure_name = item

        if self.structure_mapping is not None:
            for standardised_name in self.structure_mapping:
                for variation in self.structure_mapping[standardised_name]:
                    if variation == item:
                        return standardised_name

        return structure_name

    def __getitem__(self, item):
        structure_name = self.get_mapped_structure_name(item)

        if item not in self.structure_names:
            raise KeyError(
                f"Structure name {item} not found in structure set {self.structure_set_id}."
            )

        if item in self.cache:
            return self.cache[item]

        structure_path = self.structure_set_path.joinpath(f"{structure_name}.nii.gz")

        if not structure_path.exists():
            raise FileExistsError(
                f"No structure file found for {structure_name} in structure "
                f"set {self.structure_set_id}"
            )

        result = sitk.ReadImage(str(structure_path))

        self.cache[item] = result
        return result

    def keys(self):
        return self.structure_names

    def values(self):
        return [self[s] for s in self.structure_names]

    def items(self):
        return [(s, self[s]) for s in self.structure_names]

    def get_unmapped_structures(self) -> list:
        """Get a list of structures for which no structure was found based on the mapping. If no
        mapping is being used this will always be empty.

        Returns:
            list: Names of structures that can't be found using a mapping
        """
        missing_mappings = []
        for k in self.keys():
            structure_name = self.get_mapped_structure_name(k)
            structure_path = self.structure_set_path.joinpath(
                f"{structure_name}.nii.gz"
            )
            if not structure_path.exists():
                missing_mappings.append(k)

        return missing_mappings
