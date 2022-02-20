import logging
import sys
from pathlib import Path

from pydicer.input.pacs import DICOMPACSInput
from pydicer.input.test import TestInput
from pydicer.input.web import WebInput
from pydicer.input.filesystem import FileSystemInput
from pydicer.preprocess.data import PreprocessData
from pydicer.convert.data import ConvertData
from pydicer.visualise.data import VisualiseData


logger = logging.getLogger(__name__)


def run_pipeline(input_method, *args):
    """Run the pipeline using a specific input methodthe test data provided

    Args:
        working_dir (str, optional): Path to store test data.
        input_method (str): the input method chosen to run this pipeline
    """

    logging.basicConfig(format="%(name)s\t%(levelname)s\t%(message)s", level=logging.DEBUG)

    logger.info("Running Pipeline with Test Input")
    print(args[0])
    directory = Path(args[0])
    directory.mkdir(exist_ok=True, parents=True)

    working_dir = directory.joinpath("working")
    working_dir.mkdir(exist_ok=True, parents=True)
    output_dir = directory.joinpath("output")
    output_dir.mkdir(exist_ok=True, parents=True)

    if input_method == "test":
        input_obj = testinput_cli(*args)
    if input_method == "web":
        input_obj = web_cli(*args)
    elif input_method == "pacs":
        input_obj = pacs_cli(*args)
    else:
        input_obj = FileSystemInput(*args)

    # Preprocess the data fetch to prepare it for conversion
    logger.info("Running Pipeline preprocessing")
    preprocessed_data = PreprocessData(input_obj.working_directory, output_dir)
    preprocessed_result = preprocessed_data.preprocess()

    # Convert the data into the output directory
    logger.info("Running Pipeline conversion")
    convert_data = ConvertData(preprocessed_result, output_directory=output_dir)
    convert_data.convert()

    # TODO Visualise the converted data
    logger.info("Running Pipeline visualisation")
    visualise_data = VisualiseData(output_dir)
    visualise_data.visualise()


def testinput_cli(working_dir):
    """Trigger the test input as a mini pipeline for the CLI tool

    Example usage:
        python -m pydicer.cli.run input --type test ./cli_test

    Args:
        working_dir (str|pathlib.Path, optional): The working directory in which to
        store the data fetched.
    """
    logging.basicConfig(format="%(name)s\t%(levelname)s\t%(message)s", level=logging.DEBUG)

    logger.info("Running Test Input sub command")
    test_input = TestInput(working_dir)
    test_input.fetch_data()
    return test_input


def pacs_cli(
    working_dir,
    host="www.dicomserver.co.uk",
    port=11112,
    ae_title=None,
    modalities="GM",
    *patients
):
    """Trigger the DICOM PACS input as a mini pipeline for the CLI tool. If no inputs received,
    then by default it will retrieve some test data

    Example usage:
        python -m pydicer.cli.run input --type pacs ./cli_test www.dicomserver.co.uk 11112 DCMQUERY
            GM PAT004 PAT005

    Args:
        working_dir (str|pathlib.Path, optional): The working directory in which to
            store the data fetched.
        host (str, optional): The IP address of host name of DICOM PACS. Defaults to
            "www.dicomserver.co.uk".
        port (int, optional): The port to use to communicate on. Defaults to 11112.
        ae_title (str, optional): AE Title to provide the DICOM service. Defaults to None.
        modalities (str, optional): The modalities to retrieve DICOMs for. Defaults to "GM".
        patients (str, required): a string-list of patient IDs (IDs seperated by spaces) to
            retrieve the DICOMs for.
    """
    if not patients:
        logger.error(
            "No patient IDs provided, please provided a list-string separated by spaces of "
            "patients IDs to query for "
        )
        sys.exit()
    logger.info("Running DICOM PACS Input sub command")
    pacs_input = DICOMPACSInput(host, int(port), ae_title, working_dir)
    pacs_input.fetch_data(patients, [modalities])
    return pacs_input


def tcia_cli():
    """Trigger the TCIA input as a mini pipeline for the CLI tool."""
    return


def web_cli(working_dir, data_url):
    """Trigger the web input as a mini pipeline for the CLI tool.

    Example usage:
        python -m pydicer.cli.run input --type web ./cli_test
            https://zenodo.org/record/5276878/files/HNSCC.zip

    Args:
        working_dir (str|pathlib.Path): The working directory in which to
            store the data fetched.
        data_url (str): URL of the dataset to be downloaded from the internet
    """

    logger.info("Running web Input sub command")
    web_input = WebInput(data_url, working_dir)
    web_input.fetch_data()
    return web_input
