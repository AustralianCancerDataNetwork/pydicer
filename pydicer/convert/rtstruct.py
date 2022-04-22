import logging
from pathlib import Path
import pydicom
import SimpleITK as sitk
from matplotlib import cm

from platipy.dicom.io.rtstruct_to_nifti import transform_point_set_from_dicom_struct
from platipy.imaging.utils.io import write_nrrd_structure_set

logger = logging.getLogger(__name__)


def convert_rtstruct(
    dcm_img,
    dcm_rt_file,
    prefix="Struct_",
    output_dir=".",
    output_img=None,
    spacing=None,
):
    """Convert a DICOM RTSTRUCT to NIFTI masks.

    The masks are stored as NIFTI files in the output directory

    Args:
        dcm_img (list|SimpleITK.Image): List of DICOM paths (as str) to use as the reference image
            series or a SimpleITK image of the already converted image.
        dcm_rt_file (str|pathlib.Path): Path to the DICOM RTSTRUCT file
        prefix (str, optional): The prefix to give the output files. Defaults to "Struct" +
            underscore.
        output_dir (str|pathlib.Path, optional): Path to the output directory. Defaults to ".".
        output_img (str|pathlib.Path, optional): If set, write the reference image to this file as
            in NIFTI format. Defaults to None.
        spacing (list, optional): Values of image spacing to override. Defaults to None.
    """

    logger.debug("Converting RTStruct: %s", dcm_rt_file)
    logger.debug("Output file prefix: %s", prefix)

    if isinstance(dcm_img, list):
        dicom_image = sitk.ReadImage(dcm_img)
    elif isinstance(dcm_img, sitk.Image):
        dicom_image = dcm_img
    else:
        raise ValueError("dcm_img must be list or SimpleITK.Image")

    dicom_struct = pydicom.read_file(dcm_rt_file, force=True)

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    image_output_path = None
    if output_img is not None:

        if not isinstance(output_img, Path):
            if not output_img.endswith(".nii.gz"):
                output_img = f"{output_img}.nii.gz"
            output_img = output_dir.joinpath(output_img)

        image_output_path = output_img
        logger.debug("Image series to be converted to: %s", image_output_path)

    if spacing:

        if isinstance(spacing, str):
            spacing = [float(i) for i in spacing.split(",")]
        logger.debug("Overriding image spacing with: %s", spacing)

    struct_list, struct_name_sequence = transform_point_set_from_dicom_struct(
        dicom_image, dicom_struct, spacing
    )

    for struct_index, struct_image in enumerate(struct_list):
        out_name = f"{prefix}{struct_name_sequence[struct_index]}.nii.gz"
        out_name = output_dir.joinpath(out_name)
        logger.debug("Writing file to: %s", out_name)
        sitk.WriteImage(struct_image, str(out_name))

    if image_output_path is not None:
        sitk.WriteImage(dicom_image, str(image_output_path))


def write_nrrd_from_mask_directory(mask_directory, output_file, colormap=cm.get_cmap("rainbow")):
    """Produce a NRRD file from a directory of masks in Nifti format

    Args:
        mask_directory (pathlib.Path|str): Path object of directory containing masks
        output_file (pathlib.Path|str): The output NRRD file to write to.
        color_map (matplotlib.colors.Colormap | dict, optional): Colormap to use for output.
            Defaults to cm.get_cmap("rainbow").
    """

    if isinstance(mask_directory, str):
        mask_directory = Path(mask_directory)

    masks = {
        p.name.replace(".nii.gz", ""): sitk.ReadImage(str(p))
        for p in mask_directory.glob("*.nii.gz")
    }

    write_nrrd_structure_set(masks, output_file=output_file, colormap=colormap)
    logger.debug("Writing NRRD Structure Set to: %s", output_file)
