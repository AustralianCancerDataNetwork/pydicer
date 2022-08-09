from pathlib import Path
import logging
from pydicer.config import PyDicerConfig

from pydicer.input.base import InputBase
from pydicer.preprocess.data import PreprocessData
from pydicer.convert.data import ConvertData
from pydicer.utils import read_preprocessed_data
from pydicer.visualise.data import VisualiseData
from pydicer.dataset.preparation import PrepareDataset
from pydicer.analyse.data import AnalyseData

logger = logging.getLogger(__name__)


class PyDicer:
    def __init__(self, working_directory="."):

        self.working_directory = Path(working_directory)
        self.pydicer_directory = self.working_directory.joinpath(".pydicer")

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

        # TODO Define logging into pydicer directory

        self.convert = ConvertData(self.working_directory)
        self.visualise = VisualiseData(self.working_directory)
        self.dataset = PrepareDataset(self.working_directory)
        self.analyse = AnalyseData(self.working_directory)

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
            self.dicom_directories.append(Path(input_obj))
        elif isinstance(input_obj, InputBase):
            self.dicom_directories.append(Path(input_obj.working_directory))
        else:
            raise ValueError("input_obj must be of type str, pathlib.Path or inherit InputBase")

    def preprocess(self, force=True):
        """Preprocess the DICOM data in preparation for conversion

        Args:
            force (bool, optional): When True, all DICOM data will be re-processed (even if it has
                already been preprocessed). Defaults to True.
        """

        if len(self.dicom_directories) == 0:
            raise ValueError("No DICOM input locations set. Add one using the add_input function.")

        if self.pydicer_directory.joinpath("preprocessed.csv").exists() and not force:
            logger.debug("Data already preprocessed")
            self.preprocessed_data = read_preprocessed_data(self.working_directory)
            return

        pd = PreprocessData(self.working_directory)
        pd.preprocess(self.dicom_directories)

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

        self.analyse.compute_radiomics(dataset_name="data", patient=patient, force=force)
        self.analyse.compute_dvh(dataset_name="data", patient=patient, force=force)

    # Object generation (insert in dataset(s) or all data)
    def add_object_to_dataset(
        self,
        uid,
        patient_id,
        obj_type,
        modality,
        for_uid=None,
        referenced_sop_instance_uid=None,
        datasets=None,
    ):
        """_summary_

        Args:
            uid (_type_): _description_
            patient_id (_type_): _description_
            obj_type (_type_): _description_
            modality (_type_): _description_
            for_uid (_type_, optional): _description_. Defaults to None.
            referenced_sop_instance_uid (_type_, optional): _description_. Defaults to None.
            datasets (_type_, optional): _description_. Defaults to None.
        """

        # Check that object folder exists, if not provide instructions for adding

        # Check that no object with uid already exists

        # Check that references sop uid exists, only warning if not

        # Once ready, add to converted.csv for each dataset specified
