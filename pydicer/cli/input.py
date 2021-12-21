import sys

from pydicer.input.test import TestInput
from pydicer.input.pacs import DICOMPACSInput


def testinput_cli(working_dir=None):
    """Trigger the test input as a mini pipiline for the CLI tool

    Args:
        working_dir (str|pathlib.Path, optional): The working directory in which to
        store the data fetched. Defaults to a temp directory.
    """
    print("Running Test Input sub command")
    test_input = TestInput(working_dir)
    test_input.fetch_data()


def filesystem_cli():
    pass


def pacs_cli(
    host="www.dicomserver.co.uk",
    port=11112,
    ae_title=None,
    working_dir=None,
    modalities="GM",
    *patients
):
    """Trigger the DICOM PACS input as a mini pipiline for the CLI tool. If no inputs received,
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
            store the data fetched. Defaults to a temp directory. Defaults to None.
        modalities (str, optional): The modalities to retrieve DICOMs for. Defaults to "GM".
        patients (str, required): a string-list of patient IDs (separated with spaces between each
        element) to retrieve the DICOMs for.
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
    pass


def web_cli():
    pass
