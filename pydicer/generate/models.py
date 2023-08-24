import tempfile
import logging
from pathlib import Path

import SimpleITK as sitk

logger = logging.getLogger(__name__)


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
        for mask_file in output_dir.glob("*.nii.gz"):
            mask = sitk.ReadImage(str(mask_file))

            structure_name = mask_file.name.replace(".nii.gz")

            # Check if the mask is empty, total segmentator stores empty mask files for structures
            # that aren't within FOV
            if sitk.GetArrayFromImage(mask).sum() == 0:
                logger.debug("Segmentation mask for %s is empty, skipping...", structure_name)
                continue

            logger.debug("Loading segmentation mask for %s", structure_name)
            results[structure_name] = mask

    logger.debug("TotalSegmentator complete")

    return results
