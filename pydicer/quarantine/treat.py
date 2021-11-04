from pathlib import Path
import shutil

import pandas as pd


def copy_file_to_quarantine(file, output_directory, error_msg):
    """Move a DICOM file that couldn't be processed into the quarantine directory

    Args:
        file (Path): DICOM path to be moved into quarantine
        working_directory (Path): The working directory in which the data is stored (Output of the
        Input module)
        error_msg (str): error message associated with the quarantined file
    """
    quaran_dir = Path(output_directory).joinpath("quarantine")
    file_dir = quaran_dir.joinpath(str(file.parents[0]))
    # Create "quarantine/PATH_TO_DCM" directory
    file_dir.mkdir(exist_ok=True, parents=True)
    # Copy original DCM file to quarantine area
    shutil.copy(str(file), file_dir)

    # Create (if doesn't exist) summary file to hold info about file error
    quaran_dir.touch("summary.csv")
    summary_file = quaran_dir.joinpath("summary.csv")
    with open(summary_file, "a", encoding="utf8") as f:
        f.write(f"{str(file)}, {error_msg}\n")


class TreatImages:
    """
    Class to treat the quarantined images and prepare it for further processing

    Args:
        quaran_directory (Path): path to the quarantine directory
    """

    def __init__(self, quaran_directory):
        self.quaran_directory = quaran_directory

    def treat_images(self):
        summary_df = pd.read_csv(self.quaran_directory.joinpath("summary.csv"))
        print(summary_df)
