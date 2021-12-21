import sys

from pydicer.input.pacs import DICOMPACSInput
from pydicer.input.test import TestInput
from pydicer.input.web import WebInput


def testinput_cli(working_dir=None):
    """Trigger the test input as a mini pipeline for the CLI tool

    Args:
        working_dir (str|pathlib.Path, optional): The working directory in which to
        store the data fetched. Defaults to a temp directory.
    """
    print("Running Test Input sub command")
    test_input = TestInput(working_dir)
    test_input.fetch_data()


def pacs_cli(
    host="www.dicomserver.co.uk",
    port=11112,
    ae_title=None,
    working_dir=None,
    modalities="GM",
    *patients
):
    """Trigger the DICOM PACS input as a mini pipeline for the CLI tool. If no inputs received,
    then by default it will retrieve some test data

    Example usage:
        python -m pydicer.cli.run input --type pacs www.dicomserver.co.uk 11112 DCMQUERY
            cli_test GM PAT004 PAT005

    Args:
        host (str, optional): The IP address of host name of DICOM PACS. Defaults to
            "www.dicomserver.co.uk".
        port (int, optional): The port to use to communicate on. Defaults to 11112.
        ae_title (str, optional): AE Title to provide the DICOM service. Defaults to None.
        working_dir (str|pathlib.Path, optional): The working directory in which to
            store the data fetched. Defaults to a temp directory.
        modalities (str, optional): The modalities to retrieve DICOMs for. Defaults to "GM".
        patients (str, required): a string-list of patient IDs (IDs seperated by spaces) to
            retrieve the DICOMs for.
    """
    if not patients:
        print(
            "No patient IDs provided, please provided a list-string separated by spaces of "
            "patients IDs to query for "
        )
        sys.exit()
    print("Running DICOM PACS Input sub command")
    pacs_input = DICOMPACSInput(host, int(port), ae_title, working_dir)
    pacs_input.fetch_data(patients, [modalities])


def tcia_cli():
    """Trigger the TCIA input as a mini pipeline for the CLI tool."""
    return


def web_cli(data_url, working_dir=None):
    """Trigger the web input as a mini pipeline for the CLI tool.

    Example usage:
        python -m pydicer.cli.run input --type web
            https://zenodo.org/record/5276878/files/HNSCC.zip ./cli_test

    Args:
        data_url (str): URL of the dataset to be downloaded from the internet
        working_dir (str|pathlib.Path, optional): The working directory in which to
            store the data fetched. Defaults to a temp directory.
    """

    print("Running web Input sub command")
    web_input = WebInput(data_url, working_dir)
    web_input.fetch_data()
