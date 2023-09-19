import tempfile
import logging
from pathlib import Path

import SimpleITK as sitk

logger = logging.getLogger(__name__)


def load_output_nifti(output_dir: Path) -> dict:
    """Loads segmentation masks saved as Nifti's in an output directory into a dictionary for use
    in PyDicer.

    Args:
        output_dir (Path): The output directory of a segmentation model.

    Returns:
        dict: Dictionary of segmentation masks with the structure name as key and sitk.Image mask
            as value.
    """
    results = {}

    # Load the output masks into a dict to return
    for mask_file in output_dir.glob("*.nii.gz"):
        mask = sitk.ReadImage(str(mask_file))

        structure_name = mask_file.name.replace(".nii.gz", "")

        # Check if the mask is empty, total segmentator stores empty mask files for structures
        # that aren't within FOV
        if sitk.GetArrayFromImage(mask).sum() == 0:
            logger.debug("Segmentation mask for %s is empty, skipping...", structure_name)
            continue

        logger.debug("Loading segmentation mask for %s", structure_name)
        results[structure_name] = mask

    return results


def run_total_segmentator(input_image: sitk.Image) -> dict:
    """Run Total Segmentator on a given input image. Ensure the Total Segmentator is installed:

    ```
    pip install TotalSegmentator
    ```

    See https://github.com/wasserth/TotalSegmentator for more information.

    Args:
        input_image (sitk.Image): Input image (should be CT) to segment.

    Returns:
        dict: Dictionary of segmentations with structure name as key and sitk.Image mask as value.
    """

    # Import within function since this is an optional dependency
    # pylint: disable=import-outside-toplevel
    from totalsegmentator.python_api import totalsegmentator

    results = {}

    with tempfile.TemporaryDirectory() as working_dir:
        logger.debug("Running TotalSegmentator in temporary directory: %s", working_dir)

        working_dir = Path(working_dir)

        # Save the temporary image file for total segmentator to find
        input_dir = working_dir.joinpath("input")
        input_dir.mkdir()
        input_file = input_dir.joinpath("img.nii.gz")
        sitk.WriteImage(input_image, str(input_file))

        # Prepare a temporary folder for total segmentator to store the output
        output_dir = working_dir.joinpath("output")
        output_dir.mkdir()

        # Run total segmentator
        totalsegmentator(input_file, output_dir)

        # Load the output masks into a dict to return
        results = load_output_nifti(output_dir)

    logger.debug("TotalSegmentator complete")

    return results


def get_available_mhub_models() -> dict:
    """Determine which mHub models have been configured for use in PyDicer.

    Returns:
        dict: A dictionary with mhub model id as key and the path to the config file as value.
    """

    available_models = {}
    model_config_directory = Path(__file__).parent.joinpath("mhubconfigs")
    logger.debug("Loading mHub model configs from %s", model_config_directory)
    for model_config in model_config_directory.glob("*.yml"):
        available_models[model_config.name.replace(".yml", "")] = model_config.absolute()

    logger.debug("Found available configs: %s", available_models)
    return available_models


def run_mhub_model(
    input_image: sitk.Image,
    mhub_model: str,
    mhub_config_file: Path = None,
    gpu: bool = True,
) -> dict:
    """Use Docker to run a model made available through mHub: https://mhub.ai/

    Args:
        input_image (sitk.Image): The SimpleITK image to segment.
        mhub_model (str): The name of the model to run. Must be configured
            (check `get_available_mhub_models`) or a custom mhub_config_file should be provided.
        mhub_config_file (Path, optional): Path to a custom config file to use. Defaults to None.
        gpu (bool, optional): If True, all gpus will be requested when running the Docker image.
            Defaults to True.

    Raises:
        ImportError: Raised if the Python Docker SDK is not installed.
        ValueError: Raised if an mHub model which has not been configured for use in PyDicer is
            requested. Use the `get_available_mhub_models` function to determine available models.

    Returns:
        dict: Dictionary of segmentations with structure name as key and sitk.Image mask as value.
    """

    try:
        # pylint: disable=import-outside-toplevel
        import docker
    except ImportError as ie:
        raise ImportError(
            "Docker Python package is required to run mHub models. Install with: "
            "pip install docker"
        ) from ie

    client = docker.from_env()

    mhub_image = f"mhubai/{mhub_model}"

    # Try pulling the image
    try:
        client.images.pull(mhub_image)
    except docker.errors.ImageNotFound as inf:
        raise docker.errors.ImageNotFound(
            f"The mhub image {mhub_image} could not be pulled. "
            "Check if this model is available using the get_available_mhub_models function."
        ) from inf

    if mhub_config_file is None:
        available_mhub_models = get_available_mhub_models()

        if not mhub_model in available_mhub_models:
            raise ValueError(f"mHub model {mhub_model} not configured for use in PyDicer.")

        mhub_config_file = available_mhub_models[mhub_model]

    with tempfile.TemporaryDirectory() as working_dir:
        logger.info("Running mHub model %s in temporary %s", mhub_model, working_dir)
        working_dir = Path(working_dir)
        input_dir = working_dir.joinpath("input")
        input_dir.mkdir()
        input_file = input_dir.joinpath("image.nii.gz")
        sitk.WriteImage(input_image, str(input_file))

        output_dir = working_dir.joinpath("output")
        output_dir.mkdir()

        device_requests = []
        if gpu:
            # Request all GPUs
            device_requests = [docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])]

        volumes = {
            input_dir.absolute(): {"bind": "/app/data/input_data", "mode": "rw"},
            output_dir.absolute(): {"bind": "/app/data/output_data", "mode": "rw"},
            mhub_config_file: {"bind": "/app/data/config.yml", "mode": "rw"},
        }

        client.containers.run(
            mhub_image,
            command="--config /app/data/config.yml",
            remove=True,
            volumes=volumes,
            device_requests=device_requests,
        )

        # Load the output masks into a dict to return
        results = load_output_nifti(output_dir)

    logger.debug("mHub segmentation complete")

    return results
