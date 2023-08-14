import os
import logging
from pathlib import Path
from typing import Union, List

import SimpleITK as sitk
import pandas as pd
from sklearn.model_selection import train_test_split

from platipy.imaging.label.utils import correct_volume_overlap

from pydicer.constants import CONVERTED_DIR_NAME, DEFAULT_MAPPING_ID
from pydicer.utils import read_converted_data
from pydicer.dataset.structureset import StructureSet

logger = logging.getLogger(__name__)


class NNUNetDataset:
    working_directory = None
    dataset_name = CONVERTED_DIR_NAME
    nnunet_raw_path = None

    dataset_id = None
    dataset_name = None
    dataset_description = ""

    image_modality = None

    structure_names = None

    training_cases = []
    testing_cases = []

    assign_overlap_to_largest = False

    def __init__(
        self,
        working_directory: Union[str, Path],
        nnunet_id: int,
        nnunet_name: str,
        nnunet_description: str = "",
        dataset_name: str = CONVERTED_DIR_NAME,
        image_modality: str = "CT",
        mapping_id=DEFAULT_MAPPING_ID,
    ):
        """_summary_

        Args:
            working_directory (Union[str, Path]): _description_
            nnunet_id (int): _description_
            nnunet_name (str): _description_
            nnunet_description (str, optional): _description_. Defaults to "".
            dataset_name (str, optional): _description_. Defaults to CONVERTED_DIR_NAME.
            image_modality (str, optional): _description_. Defaults to "CT".
            mapping_id (str, optional): _description_. Defaults to DEFAULT_MAPPING_ID.

        Raises:
            SystemError: _description_
        """

        # Check that the nnUNet_raw_data_base environment variable is set
        if not "nnUNet_raw" in os.environ:
            raise SystemError(
                "'nnUNet_raw' environment variable not set. "
                "Ensure nnUNet has been properly configured before continuing."
            )

        self.nnunet_raw_path = Path(os.environ["nnUNet_raw"])

        self.working_directory = working_directory
        self.nnunet_id = nnunet_id
        self.nnunet_name = nnunet_name
        self.nnunet_description = nnunet_description
        self.dataset_name = dataset_name
        self.image_modality = image_modality
        self.mapping_id = mapping_id

    def check_dataset(self):
        """Check to see that this dataset has been prepared properly.

        Expect to have exactly 1 image of the modality configured and 1 structure set.
        """
        df = read_converted_data(self.working_directory, dataset_name=self.dataset_name)

        flags = []
        for patient_id, df_pat in df.groupby("patient_id"):
            image_count = len(df_pat[df_pat.modality == self.image_modality])
            structure_count = len(df_pat[df_pat.modality == "RTSTRUCT"])

            if image_count != 1:
                flags.append(
                    f"Found {image_count} {self.image_modality} images for {patient_id}. "
                    "Expected exactly 1 image."
                )

            if structure_count != 1:
                flags.append(
                    f"Found {structure_count} structure sets for {patient_id}. "
                    "Expected exactly 1 structure set."
                )

        if len(flags) > 0:
            for flag in flags:
                logger.error(flag)

            raise SystemError(
                "Dataset has not been prepared. Use the dataset preparation module "
                "to prepare a datset with exactly 1 {self.modality} image and 1 "
                "structure set."
            )

        logger.info("Dataset OK")

    def split_dataset(
        self,
        training_cases: List[str] = None,
        testing_cases: List[str] = None,
        patients: List[str] = None,
        **kwargs,
    ):
        """Split the dataset by either supplying the training and testing cases. If these are not
        supplied a split will be done using sklearn's train_test_split. Key-word arguments passed
        through to this function will be passed on to train_test_split.

        Args:
            training_cases (List[str], optional): Specify a list of training cases, won't split
                using train_test_split function in this scenario. Defaults to None.
            testing_cases (List[str], optional): Specify list of testing cases, can only be
                supplied if training_cases is also supplied. Defaults to None.
            patients (List[str], optional): Define a subset of patient to use for train_test_split.
                If None then all patients wil be used. Defaults to None.

        Raises:
            AttributeError: Raised when testing_cases is set but training_cases is not.
            ValueError: Raised when training or testing case not present in dataset.
        """

        df = read_converted_data(self.working_directory, dataset_name=self.dataset_name)
        patient_list = df.patient_id.unique().tolist()

        if training_cases is None:
            # No training cases specified, will do a split using sklearn.
            # Check that no testing cases were set in this scenario.
            if training_cases is not None:
                raise AttributeError(
                    "testing_cases supplied but training_cases not supplied. If "
                    "supplying testing_cases ensure that training_cases are also "
                    "supplied."
                )

            if patients is not None:
                patient_list = patients

            # Set a default random state if one is not already supplied
            if "random_state" not in kwargs:
                kwargs["random_state"] = 42

            training_cases, testing_cases = train_test_split(patient_list, **kwargs)

        if testing_cases is None:
            # If training cases were set but testing cases were not we'll end up here
            # No problem, testing cases are optional, just set to empty list
            testing_cases = []

        # Sanity check, make sure we have all our training and testing cases in our dataset
        for case in training_cases:
            if not case in patient_list:
                raise ValueError(f"Training case {case} not found in dataset.")
        for case in testing_cases:
            if not case in patient_list:
                raise ValueError(f"Testing case {case} not found in dataset.")

        self.training_cases = training_cases
        self.testing_cases = testing_cases

        logger.info("Dataset split OK")
        logger.info("Training cases: %s", self.training_cases)
        logger.info("Testing cases: %s", self.testing_cases)

    def add_testing_cases(self, testing_cases: List[str]):
        """Add some testing cases only. Can be useful if wanting to analyse more data after a model
        has been trained.

        Args:
            testing_cases (list): A list of case IDs to add to the training set.
        """

        df = read_converted_data(self.working_directory, dataset_name=self.dataset_name)
        patient_list = df.patient_id.unique().tolist()
        for case in testing_cases:
            if not case in patient_list:
                raise ValueError(f"Testing case {case} not found in dataset.")

        self.testing_cases += testing_cases

    def check_duplicates_train_test(self):
        """Check the images in the train and test sets to determine if there are any inadvertant
        duplicates.

        This can be useful since sometimes when datasets are anonymised multiple times the same
        dataset might have a different anonymised patients ID. Best to find this out before
        training the model so that these cases can be removed from the training or testing set.

        Raises:
            SystemError: Raised if `split_dataset` has not yet been run
        """

        if len(self.training_cases) == 0:
            raise SystemError("training_cases are empty, run split_dataset function first.")

        img_stats = []

        df = read_converted_data(self.working_directory, dataset_name=self.dataset_name)
        df_images = df[(df.modality == "CT") | (df.modality == "MR") | (df.modality == "PT")]

        for case in self.training_cases + self.testing_cases:
            df_pat = df_images[df_images.patient_id == case]

            for _, img_row in df_pat.iterrows():
                img_path = Path(img_row.path).joinpath(f"{img_row.modality}.nii.gz")
                img = sitk.ReadImage(str(img_path))

                spacing = img.GetSpacing()
                size = img.GetSize()

                img_stats.append(
                    {
                        "case": case,
                        "hashed_uid": img_row.hashed_uid,
                        "img_path": str(img_path),
                        "modality": img_row.modality,
                        "set": "training" if case in self.training_cases else "testing",
                        "spacing": spacing,
                        "size": size,
                    }
                )

        df_img_stats = pd.DataFrame(img_stats)

        # Check to see if we have any duplicate image spacing and sizes, if so inspect these
        # further
        duplicated_rows = df_img_stats.duplicated(subset=["spacing", "size"], keep=False)
        df_duplicated = df_img_stats[duplicated_rows]
        df_duplicated.loc[duplicated_rows, "voxel_sum"] = df_duplicated.img_path.apply(
            lambda img_path: sitk.GetArrayFromImage(sitk.ReadImage(img_path)).sum()
        )

        duplicates_found = False
        for _, df_group in df_duplicated.groupby("voxel_sum"):
            if len(df_group) == 1:
                continue

            duplicates_found = True

            first_row = df_group.iloc[0]
            for idx in range(1, len(df_group)):
                row = df_group.iloc[idx]
                log_msg = (
                    f"Image {first_row.hashed_uid} for case {first_row.case} is likely a "
                    f"duplicate of image {row.hashed_uid} for case {row.case}"
                )

                if first_row.set == row.set:
                    logger.warning(log_msg)
                else:
                    logger.error(log_msg)

        if not duplicates_found:
            logger.info("No duplicate images found in training and testing sets")

    def check_structure_names(self) -> pd.DataFrame:
        """Prepare a DataFrame to indicate which structures are available/missing for each patient
        in the dataset.

        Returns:
            pd.DataFrame: DataFrame indicating structures available.
        """

        df = read_converted_data(self.working_directory, dataset_name=self.dataset_name)
        df_structure_sets = df[df.modality == "RTSTRUCT"]

        # First get a set of all unique structure names available
        structures_available = set()
        for _, row in df_structure_sets.iterrows():
            structure_set = StructureSet(row, mapping_id=self.mapping_id)
            set_structure_names = set(structure_set.structure_names)
            structures_available = set.union(structures_available, set_structure_names)

        structures_available = sorted(structures_available)

        # Prepare a DataFrame to display which cases have which structures available
        results = []
        for _, row in df_structure_sets.iterrows():
            case_details = {"patient_id": row.patient_id, "struct_hash": row.hashed_uid}

            structure_set = StructureSet(row, mapping_id=self.mapping_id)
            unmapped_structures = structure_set.get_unmapped_structures()

            struct_details = {
                k: 0 if k in unmapped_structures else 1 for k in structures_available
            }
            results.append({**case_details, **struct_details})

        df_results = pd.DataFrame(results)

        df_return = df_results.style.apply(
            lambda col: [
                "background-color: red" if c == 0 else "background-color: green"
                for c in col.values
            ],
            subset=structures_available,
        )

        self.structure_names = structures_available

        # If anything is missing, advise the user what is missing and what to do
        num_patients = len(df_results)
        summed_results = df_results.sum()
        incomplete_structures = []
        incomplete_patients = []
        import matplotlib

        cm = matplotlib.colormaps.get_cmap("rainbow")
        cm = matplotlib.colors.Colormap("test")
        cm = cm.resampled(10)
        for s in self.structure_names:
            if not num_patients == summed_results[s]:
                missing_pats = df_results[df_results[s] == 0].patient_id.tolist()
                print(f"Structure {s} is missing for patients: {missing_pats}")

                incomplete_structures.append(s)
                incomplete_patients += [p for p in missing_pats if not p in incomplete_patients]

        if incomplete_structures:
            print(
                "Not all structures available for all patients. Consider removing structures "
                f"from nnUNet (by removing these from mapping): {incomplete_structures}"
            )

        if incomplete_patients:
            print(
                "Not all patient datasets have all structures available. Consider removing these "
                f"patients from the dataset: {incomplete_patients}"
            )

        return df_return

    def check_overlapping_structures(self):
        """Determine if any of the structures are overlapping. The nnUNet does not support
        overlapping structures. If any overlapping structures exist voxels will be assigned to the
        smallest structure by default or to the largest structure if `assign_overlap_to_largest` is
        True."""

        if self.structure_names is None:
            logger.error(
                "Structures not ready, run check_structure_names before checking "
                "overlapping structures."
            )
        df = read_converted_data(self.working_directory, dataset_name=self.dataset_name)
        df_structure_sets = df[df.modality == "RTSTRUCT"]

        has_overlapping_structures = False
        for _, row in df_structure_sets.iterrows():
            structure_set = StructureSet(row, mapping_id=self.mapping_id)

            structure_names = structure_set.keys()
            for si, structure_name_i in enumerate(structure_names):
                for sj in range(si + 1, len(structure_names)):
                    structure_name_i = structure_names[si]
                    structure_name_j = structure_names[sj]

                    structure_sum = (
                        structure_set[structure_name_i] + structure_set[structure_name_j]
                    )
                    arr = sitk.GetArrayFromImage(structure_sum)
                    if arr.max() > 1:
                        print(
                            f"{structure_name_i} overlaps with {structure_name_j} for patient "
                            f"{row.patient_id} structure set {row.struct_hash}"
                        )
                        has_overlapping_structures = True

        if has_overlapping_structures:
            logger.warning("Overlapping structures were detected")

    def prep_label_map_from_one_hot(
        self, image: sitk.Image, structure_set: StructureSet
    ) -> sitk.Image:
        """Prepare a label map from a structure set. Since overlapping structures aren't supported
        in a label map, voxels will be assigned to the larger structure if
        `assign_overlap_to_largest` is True or the smaller structure if `assign_overlap_to_largest`
        is False.

        Args:
            image (sitk.Image): The image corresponding to the structure set.
            structure_set (StructureSet): The structure set from which to create the label map.

        Returns:
            sitk.Image: The label map.
        """

        # correct overlap
        structures = correct_volume_overlap(
            structure_set, assign_overlap_to_largest=self.assign_overlap_to_largest
        )

        # create the label map
        label_map = None
        for enum, s in enumerate(self.structure_names):
            mask = structures[s]

            if label_map is None:
                label_map = mask
            else:
                label_map = label_map + (1 + enum) * mask

        return sitk.Resample(label_map, image)

    def prepare_dataset(self):
        pass

    def generate_training_scripts(self):
        pass
