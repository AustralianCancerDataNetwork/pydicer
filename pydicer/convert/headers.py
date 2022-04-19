import logging
import json
import pydicom

logger = logging.getLogger(__name__)


def convert_dicom_headers(dcm_file, binary_path, json_file):
    """Save the DICOM Headers as a JSON file

    Args:
        dcm_file (str|pathlib.Path): The files from which to save the headers.
        binary_path (str): Relative path to binary data which will be placed into JSON.
        json_file (str|pathlib.Path): Path to JSON file to save output.
    """

    # Write the DICOM headers (of the first slice) to JSON
    dcm_ds = pydicom.read_file(dcm_file, force=True)
    dcm_dict = dcm_ds.to_json_dict(
        bulk_data_threshold=4096, bulk_data_element_handler=lambda _: binary_path
    )

    with open(json_file, "w", encoding="utf8") as jsonfile:
        json.dump(dcm_dict, jsonfile, indent=2)

    logger.debug("DICOM Headers written to: %s", json_file)
