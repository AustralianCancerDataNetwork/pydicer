import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

from pydicer.config import PyDicerConfig
from pydicer.constants import CONVERTED_DIR_NAME, PYDICER_DIR_NAME

from pydicer.input.base import InputBase
from pydicer.preprocess.data import PreprocessData
from pydicer.convert.data import ConvertData
from pydicer.utils import read_preprocessed_data
from pydicer.visualise.data import VisualiseData
from pydicer.dataset.preparation import PrepareDataset
from pydicer.analyse.data import AnalyseData

from pydicer.utils import (
    read_converted_data,
    add_structure_name_mapping,
    get_structures_linked_to_dose,
    copy_doc,
)
from pydicer.quarantine import read_quarantined_data

from pydicer.generate.object import (
    add_object,
    add_image_object,
    add_structure_object,
    add_dose_object,
)
from pydicer.generate.segmentation import (
    read_all_segmentation_logs,
    segment_image,
    segment_dataset,
)

logger = logging.getLogger()


class PyDicer:
    """The PyDicer class provides easy access to all the key PyDicer functionality.

    Args:
        working_directory (str|pathlib.Path, optional): Directory in which data is stored. Defaults
          to ".".

    :ivar convert: Instance of :class:`~pydicer.convert.data.ConvertData`
    :ivar visualise: Instance of :class:`~pydicer.visualise.data.VisualiseData`
    :ivar dataset: Instance of :class:`~pydicer.dataset.preparation.PrepareDataset`
    :ivar analyse: Instance of :class:`~pydicer.analyse.data.AnalyseData`
    """

    def __init__(self, working_directory="."):
        self.working_directory = Path(working_directory)
        self.pydicer_directory = self.working_directory.joinpath(PYDICER_DIR_NAME)

        if self.working_directory.exists():
            # If the directory already exists, make sure it's really a PyDicer directory. If not
            # warn the user that PyDicer will be initialised in this existing directory

            if not self.pydicer_directory.exists():
                logger.warning(
                    "%s already exists but no .pydicer sub- directory was found. PyDicer will be "
                    "initialised in this directory.",
                    {self.working_directory},
                )

        self.pydicer_directory.mkdir(parents=True, exist_ok=True)

        self.dicom_directories = []

        self.preprocessed_data = None

        # Init Config
        self.config = PyDicerConfig(self.working_directory)

        # Update the loggers
        self.update_logging()

        self.convert = ConvertData(self.working_directory)
        self.visualise = VisualiseData(self.working_directory)
        self.dataset = PrepareDataset(self.working_directory)
        self.analyse = AnalyseData(self.working_directory)

    def set_verbosity(self, verbosity):
        """Set's the verbosity of the tool to the std out (console). When 0 (not set) the tool will
        display a progress bar. Other values indicate Python's build in logging levels:
        - DEBUG: 10
        - INFO: 20
        - WARNING: 30
        - ERROR: 40
        - CRITICAL = 50

        Example:
        ```python
        pd = PyDicer(working_directory)
        pd.set_verbosity(logging.INFO)
        ```

        Args:
            verbosity (int): The Python log level
        """

        self.config.set_config("verbosity", verbosity)
        self.update_logging()

    def update_logging(self):
        """Resets the loggers configured. Should be called after every config change to logging."""

        verbosity = self.config.get_config("verbosity")

        log_file_path = self.pydicer_directory.joinpath("pydicer.log")

        logger.handlers.clear()
        logger.setLevel(logging.DEBUG)

        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=100 * 1024 * 1024,  # Max 100 MB per log file before rotating
            backupCount=100,  # Keep up to 100 log files in history
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        if verbosity > 0:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                "%(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(verbosity)
            logger.addHandler(console_handler)

    def add_input(self, input_obj):
        """Add an input location containing DICOM data. Must a str, pathlib.Path or InputBase
        object, such as:
        - FileSystemInput
        - DICOMPacsInput
        - OrthancInput
        - WebInput

        Args:
            input_obj (str|pathlib.Path|InputBase): The Input object, derived from InputBase or a
              str/pathlib.Path pointing to the folder containing the DICOM files
        """

        if isinstance(input_obj, (str, Path)):
            dicom_path = Path(input_obj)
            self.dicom_directories.append(dicom_path)
            logger.debug("Added DICOM input path: %s", dicom_path)
        elif isinstance(input_obj, InputBase):
            dicom_path = Path(input_obj.working_directory)
            self.dicom_directories.append(dicom_path)
            logger.debug("Added DICOM input path: %s", dicom_path)
        else:
            raise ValueError(
                "input_obj must be of type str, pathlib.Path or inherit InputBase"
            )

    def preprocess(self, force=True):
        """Preprocess the DICOM data in preparation for conversion

        Args:
            force (bool, optional): When True, all DICOM data will be re-processed (even if it has
                already been preprocessed). Defaults to True.
        """

        if len(self.dicom_directories) == 0:
            raise ValueError(
                "No DICOM input locations set. Add one using the add_input function."
            )

        preprocess_data = PreprocessData(self.working_directory)
        preprocess_data.preprocess(self.dicom_directories, force=force)

        self.preprocessed_data = read_preprocessed_data(self.working_directory)

    def run_pipeline(self, patient=None, force=True):
        """Runs the entire conversion pipeline, including computation of DVHs and first-order
        radiomics.

        Args:
            patient (str|list, optional): A patient ID or list of patient IDs for which to run the
            pipeline. Defaults to None (Runs all patients).
            force (bool, optional): When True, all steps are re-processed even if the output files
              have previously been generated. Defaults to True.
        """

        self.preprocess(force=force)

        self.convert.convert(patient=patient, force=force)
        self.visualise.visualise(patient=patient, force=force)

        self.analyse.compute_radiomics(
            dataset_name=CONVERTED_DIR_NAME, patient=patient, force=force
        )
        self.analyse.compute_dvh(
            dataset_name=CONVERTED_DIR_NAME, patient=patient, force=force
        )

    @copy_doc(add_structure_name_mapping, remove_args=["working_directory"])
    def add_structure_name_mapping(  # pylint: disable=missing-function-docstring
        self, *args, **kwargs
    ) -> pd.DataFrame:
        return add_structure_name_mapping(
            *args, working_directory=self.working_directory, **kwargs
        )

    @copy_doc(read_preprocessed_data, remove_args=["working_directory"])
    def read_preprocessed_data(  # pylint: disable=missing-function-docstring
        self,
    ) -> pd.DataFrame:
        return read_preprocessed_data(working_directory=self.working_directory)

    @copy_doc(read_converted_data, remove_args=["working_directory"])
    def read_converted_data(  # pylint: disable=missing-function-docstring
        self, *_, **kwargs
    ) -> pd.DataFrame:
        return read_converted_data(working_directory=self.working_directory, **kwargs)

    @copy_doc(read_quarantined_data, remove_args=["working_directory"])
    # pylint: disable=missing-function-docstring
    def read_quarantined_data(self) -> pd.DataFrame:
        return read_quarantined_data(working_directory=self.working_directory)

    @copy_doc(read_quarantined_data, remove_args=["working_directory"])
    # pylint: disable=missing-function-docstring
    def get_structures_linked_to_dose(self, *args, **kwargs) -> pd.DataFrame:
        return get_structures_linked_to_dose(self.working_directory, *args, **kwargs)

    @copy_doc(add_object, remove_args=["working_directory"])
    def add_object(  # pylint: disable=missing-function-docstring
        self, *args, **kwargs
    ) -> pd.DataFrame:
        return add_object(self.working_directory, *args, **kwargs)

    @copy_doc(add_image_object, remove_args=["working_directory"])
    def add_image_object(  # pylint: disable=missing-function-docstring
        self, *args, **kwargs
    ) -> pd.DataFrame:
        return add_image_object(self.working_directory, *args, **kwargs)

    @copy_doc(add_structure_object, remove_args=["working_directory"])
    def add_structure_object(  # pylint: disable=missing-function-docstring
        self, *args, **kwargs
    ) -> pd.DataFrame:
        return add_structure_object(self.working_directory, *args, **kwargs)

    @copy_doc(add_dose_object, remove_args=["working_directory"])
    def add_dose_object(  # pylint: disable=missing-function-docstring
        self, *args, **kwargs
    ) -> pd.DataFrame:
        return add_dose_object(self.working_directory, *args, **kwargs)

    @copy_doc(read_all_segmentation_logs, remove_args=["working_directory"])
    def read_all_segmentation_logs(  # pylint: disable=missing-function-docstring
        self, *args, **kwargs
    ) -> pd.DataFrame:
        return read_all_segmentation_logs(self.working_directory, *args, **kwargs)

    @copy_doc(segment_image, remove_args=["working_directory"])
    def segment_image(  # pylint: disable=missing-function-docstring
        self, *args, **kwargs
    ) -> pd.DataFrame:
        return segment_image(self.working_directory, *args, **kwargs)

    @copy_doc(segment_dataset, remove_args=["working_directory"])
    def segment_dataset(  # pylint: disable=missing-function-docstring
        self, *args, **kwargs
    ) -> pd.DataFrame:
        return segment_dataset(self.working_directory, *args, **kwargs)
