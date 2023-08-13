import logging
import datetime
import traceback
from pathlib import Path
from typing import Callable

from tqdm import tqdm
import pandas as pd
import SimpleITK as sitk

from pydicer.constants import CONVERTED_DIR_NAME
from pydicer.utils import read_converted_data
from pydicer.generate.object import add_structure_object

logger = logging.getLogger(__name__)


def get_segmentation_log_path(image_row: pd.Series) -> Path:
    """Get the auto-segmentation log path given an image entry.

    Args:
        image_row (pd.Series): A row of an image of a converted data frame.

    Returns:
        pathlib.Path: Path to the relavant patient's auto-segmentation log.
    """

    img_path = Path(image_row.path)
    struct_path = img_path.parent.parent.joinpath("structures")
    log_path = struct_path.joinpath("segmentation_log.csv")

    return log_path


def read_segmentation_log(image_row: pd.Series) -> pd.DataFrame():
    """Read the auto-segmentation log given an image entry.

    Args:
        image_row (pd.Series): A row of an image of a converted data frame.

    Returns:
        pd.DataFrame: The pandas DataFrame object describing the log.
    """

    log_path = get_segmentation_log_path(image_row)

    col_types = {
        "patient_id": str,
        "segment_id": str,
        "img_id": str,
        "total_time_seconds": float,
        "success_flag": bool,
        "error": str,
    }

    if log_path.exists():
        return pd.read_csv(
            log_path, index_col=0, parse_dates=["start_time", "end_time"], dtype=col_types
        )

    return pd.DataFrame(
        columns=[
            "patient_id",
            "segment_id",
            "img_id",
            "start_time",
            "end_time",
            "total_time_seconds",
            "success_flag",
            "error",
        ]
    )


def write_autoseg_log(
    image_row: pd.Series,
    segment_id: str,
    start_time: datetime,
    end_time: datetime,
    success: bool,
    error_msg: str,
):
    """Writes the auto-segmentation log for an auto-segmentation run.

    Args:
        image_row (pd.Series): The row of an image of a converted data frame on which the
            auto-segmentation was run.
        segment_id (str): The ID of the auto-segmentation run.
        start_time (datetime): The date/time the run started.
        end_time (datetime): The date/time the run ended.
        success (bool): Boolean flag inicated if the auto-segmentation was run successfully or not.
        error_msg (str): A string indicating the error message if there was an error.
    """

    df = read_segmentation_log(image_row)

    entry = {
        "patient_id": image_row.patient_id,
        "segment_id": segment_id,
        "img_id": image_row.hashed_uid,
        "start_time": start_time,
        "end_time": end_time,
        "total_time_seconds": (end_time - start_time).total_seconds(),
        "success_flag": success,
        "error": error_msg,
    }

    df = pd.concat([df, pd.DataFrame([entry])])
    df = df.reset_index(drop=True)

    log_path = get_segmentation_log_path(image_row)
    df.to_csv(log_path)


def read_all_segmentation_logs(
    working_directory: Path, dataset_name: str, segment_id: str = None, modality: str = None
):
    """Read all auto-segmentation logs in a dataset.
    Args:
        dataset_name (str): The name of the dataset to read for.
        segment_id (str): The ID of the auto-segmentation run.

    Returns:
        pd.DataFrame: The pandas DataFrame object with all logs for the dataset.
    """

    df = read_converted_data(working_directory, dataset_name=dataset_name)

    df_logs = pd.DataFrame()

    if modality is not None:
        df = df[df.modality == modality]

    for _, row in df.iterrows():

        df_log = read_segmentation_log(row)

        df_logs = pd.concat([df_logs, df_log])

    df_logs = df_logs.reset_index(drop=True)

    if segment_id is not None:
        df_logs = df_logs[df_logs.segment_id == segment_id]

    return df_logs


def segment_image(
    working_directory: Path,
    image_row: pd.Series,
    segment_id: str,
    segmentation_function: Callable,
    dataset_name: str = None,
    force: bool = False,
):
    """Run an auto-segmentation function on an image. Provide the image row of the converted
    DataFrame to auto-segment, segmentation results will be save as a new object within the
    patient's data.

    The `segment_function` provided should accept a SimpleITK image as input and return a `dict`
    with structure names as keys and SimpleITK images as value.

    If you segmentation algorithm requires further customisation, consider wrapping it in a
    function to match this notation. For example, to run the TotalSegmentator, you can define
    a warpper function like:

    ```
    import tempfile

    def run_total_segmentator(img: SimpleITK.Image) -> dict:

        temp_path = Path(tempfile.mkdtemp())
        sitk.WriteImage(img, str(temp_path.joinpath("img.nii.gz")))

        // TODO fix this

    ```

    Args:
        working_directory (Path): The PyDicer working directory.
        image_row (pd.Series): The image row of the converted DataFrame to use for segmentation.
        segment_id (str): The ID to be given to track the results of this segmentation.
        segmentation_function (Callable): The function to call to run the segemtantion. Excepts a
            SimpleITK.Image as input and returns a dict object with structure names as keys and
            SimpleITK.Image masks as values.
        dataset_name (str, optional): The name of the dataset to add the segmented structure set
            to. Defaults to None.
        force (bool, optional): If True, the segmetation will be re-run. Defaults to False.

    Raises:
        TypeError: The segmentation function returned the wrong type (requies a dict)
    """

    df = read_converted_data(working_directory)

    modality = image_row.modality

    df_pat_autoseg = read_segmentation_log(image_row)

    # Check if the segmentation has already been run for this image
    segment_struct_id = f"{segment_id}_{image_row.hashed_uid}"
    print(segment_struct_id)

    # This check is to support deployments before tracking auto-seg log
    if not force and len(
        df[
            (df.referenced_sop_instance_uid == image_row.sop_instance_uid)
            & (df.hashed_uid == segment_struct_id)
        ]
    ):
        logger.info(
            "Structures already generated for patient: %s and image: %s",
            image_row.patient_id,
            image_row.hashed_uid,
        )
        return

    # Auto-seg log tracks passed and failed segmentation runs

    df_previous_runs = df_pat_autoseg[
        (df_pat_autoseg.segment_id == segment_id) & df_pat_autoseg.img_id == image_row.hashed_uid
    ]
    if not force and len(df_previous_runs) > 0:
        logger.info(
            "Auto-segmentation already run for patient: %s and image: %s",
            image_row.patient_id,
            image_row.hashed_uid,
        )
        return

    start_time = datetime.datetime.now()
    run_successful = True
    error_msg = ""

    try:
        # Compute the auto-segmentations on the image
        img = sitk.ReadImage(str(Path(image_row.path).joinpath(f"{modality}.nii.gz")))

        # Call the segmentation function depending on the additional arguments that may be needed
        segmentation_result = segmentation_function(img)

        if not isinstance(segmentation_result, dict):
            raise TypeError(
                "Segmentation function must return dict object with structure names as keys and "
                "SimpleITK.Image's as value."
            )

        # Add the object to the pydicer project directory
        add_structure_object(
            working_directory,
            segmentation_result,
            segment_struct_id,
            image_row.patient_id,
            linked_image=image_row,
            datasets=dataset_name,
        )

        # Visualise the newly added structures
        # pydicer.visualise.visualise(patient=ct_row.patient_id, force=False)

        logger.info(
            "Auto-segmentation complete for Patient: %s and image: %s",
            image_row.patient_id,
            image_row.hashed_uid,
        )

    except Exception as e:  # pylint: disable=broad-exception-caught

        run_successful = False
        error_msg = str(e)

        logger.info(
            "Auto-segmentation failed for Patient: %s and image: %s with error: %s",
            image_row.patient_id,
            image_row.hashed_uid,
            e,
        )

        error_path = working_directory.joinpath(
            CONVERTED_DIR_NAME,
            image_row.patient_id,
            "structures",
            segment_struct_id,
        )
        error_path.mkdir(parents=True, exist_ok=True)
        error_file = error_path.joinpath("auto_segment_error.log")

        # Writing error into log file
        with open(error_file, "a", encoding="utf8") as f:
            f.write(str(e))
            f.write(traceback.format_exc())

    end_time = datetime.datetime.now()

    write_autoseg_log(image_row, segment_id, start_time, end_time, run_successful, error_msg)


def segment_dataset(
    working_directory: Path,
    segment_id: str,
    segmentation_function: Callable,
    dataset_name: str = CONVERTED_DIR_NAME,
    modality: str = "CT",
    force: bool() = False,
):
    """Run an auto-segmentation function across all images of a given modality in a dataset.

    Args:
        working_directory (Path): The PyDicer working directory.
        segment_id (str): The ID to be given to track the results of this segmentation.
        segmentation_function (Callable): The function to call to run the segemtantion. Excepts a
            SimpleITK.Image as input and returns a dict object with structure names as keys and
            SimpleITK.Image masks as values.
        dataset_name (str, optional): The name of the dataset to run auto-segmentation on. Defaults
            to CONVERTED_DIR_NAME which run across all images available.
        modality (str, optional): The modality of the image to run on. Defaults to "CT".
        force (bool, optional): If True, the segmetation will be re-run for each image even if it
            was already previously run. Defaults to False.
    """

    working_directory = Path(working_directory)

    df = read_converted_data(working_directory, dataset_name=dataset_name)
    df = df[df.modality == modality]

    for _, image_row in tqdm(df.iterrows(), "Segmentation", total=len(df)):

        segment_image(
            working_directory,
            image_row,
            segment_id,
            segmentation_function,
            dataset_name=None if dataset_name == CONVERTED_DIR_NAME else dataset_name,
            force=force,
        )
