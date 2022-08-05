import logging

from pathlib import Path

from pydicer.input.test import TestInput
from pydicer.preprocess.data import PreprocessData
from pydicer.convert.data import ConvertData
from pydicer.visualise.data import VisualiseData
from pydicer.dataset.preparation import PrepareDataset

logger = logging.getLogger(__name__)


def run(input_object, working_directory=".", dicom_directory="./dicom"):
    """Run the pipeline to convert some data

    Args:
        input_object (InputBase): An object of a Class inheriting InputBase.
        output_directory (str, optional): Path in which to store converted data. Defaults to ".".
    """

    # Fetch the data from the Input Source
    input_object.fetch_data()

    # Preprocess the data fetch to prepare it for conversion
    preprocessed_data = PreprocessData(working_directory, dicom_directory)
    preprocessed_data.preprocess()

    # Convert the data into the output directory
    convert_data = ConvertData(working_directory)
    convert_data.convert()

    # Visualise the converted data
    visualise_data = VisualiseData(working_directory)
    visualise_data.visualise()

    # Dataset selection and preparation
    prepare_dataset = PrepareDataset(working_directory)
    prepare_dataset.prepare("clean", "rt_latest_struct")


def run_test(directory="./testdata"):
    """Run the pipeline using the test data provided

    Args:
        directory (str, optional): Path to store test data. Defaults to "./testdata".
    """

    logging.basicConfig(format="%(name)s\t%(levelname)s\t%(message)s", level=logging.DEBUG)

    logger.info("Running Pipeline with Test Input")
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)

    dicom_directory = directory.joinpath("dicom")
    dicom_directory.mkdir(exist_ok=True, parents=True)

    test_input = TestInput(dicom_directory)

    run(test_input, directory, dicom_directory)


if __name__ == "__main__":
    run_test()
