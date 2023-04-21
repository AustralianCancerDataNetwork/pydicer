import os
import logging
from pathlib import Path
from typing import Union, List

import SimpleITK as sitk
import pandas as pd
from sklearn.model_selection import train_test_split

from pydicer.constants import CONVERTED_DIR_NAME
from pydicer.utils import read_converted_data

logger = logging.getLogger(__name__)


class NNUNetTask:

    working_directory = None
    dataset_name = CONVERTED_DIR_NAME
    nnunet_raw_path = None

    task_id = None
    task_name = None
    task_description = ""

    image_modality = None

    structure_names = []

    training_cases = []
    testing_cases = []

    def __init__(
        self,
        working_directory: Union[str, Path],
        task_id: int,
        task_name: str,
        task_description: str = "",
        dataset_name: str = CONVERTED_DIR_NAME,
        image_modality: str = "CT",
        structure_names: List[str] = None,
    ):
        """_summary_

        Args:
            working_directory (Union[str, Path]): _description_
            task_id (int): _description_
            task_name (str): _description_
            task_description (str, optional): _description_. Defaults to "".
            dataset_name (str, optional): _description_. Defaults to CONVERTED_DIR_NAME.
            image_modality (str, optional): _description_. Defaults to "CT".
            structure_names (List[str], optional): _description_. Defaults to None.

        Raises:
            SystemError: _description_
        """

        # Check that the nnUNet_raw_data_base environment variable is set
        if not "nnUNet_raw_data_base" in os.environ:
            raise SystemError(
                "'nnUNet_raw_data_base' environment variable not set. "
                "Ensure nnUNet has been properly configured before continuing."
            )

        self.nnunet_raw_path = Path(os.environ["nnUNet_raw_data_base"]).joinpath("nnUNet_raw_data")

        self.working_directory = working_directory
        self.task_id = task_id
        self.task_name = task_name
        self.task_description = task_description
        self.dataset_name = dataset_name
        self.image_modality = image_modality
        self.structure_names = structure_names

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

    def check_structure_names(self):

        df = read_converted_data(self.working_directory, dataset_name=self.dataset_name)
        df_structure_sets = df[df.modality == "RTSTRUCT"]

        checks = []
        structures_available = set()
        for _, row in df_structure_sets.iterrows():

            structure_files = Path(row.path).glob("*.nii.gz")
            set_structure_names = set([p.name.replace(".nii.gz", "") for p in structure_files])
            structures_available = set.union(structures_available, set_structure_names)

            # checks.append({sn: sn in set_structure_names for sn in self.structure_names})

        print(structures_available)
        # print(pd.DataFrame(checks))

    def check_overlapping_structures(self):

        pass

    def prepare_dataset(self):
        pass

    def generate_training_scripts(self):
        pass
