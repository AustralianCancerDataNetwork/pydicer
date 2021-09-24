from pathlib import Path
import pydicom
import SimpleITK as sitk

from platipy.dicom.io.rtstruct_to_nifti import transform_point_set_from_dicom_struct


def convert_rtstruct(
    dcm_img_list,
    dcm_rt_file,
    prefix="Struct_",
    output_dir=".",
    output_img=None,
    spacing=None,
):
    """Convert a DICOM RTSTRUCT to NIFTI masks.

    The masks are stored as NIFTI files in the output directory

    Args:
        dcm_img_list (str|pathlib.Path): Path to the reference DICOM image series
        dcm_rt_file (str|pathlib.Path): Path to the DICOM RTSTRUCT file
        prefix (str, optional): The prefix to give the output files. Defaults to "Struct_".
        output_dir (str|pathlib.Path, optional): Path to the output directory. Defaults to ".".
        output_img (str|pathlib.Path, optional): If set, write the reference image to this file as
                                                 in NIFTI format. Defaults to None.
        spacing (list, optional): Values of image spacing to override. Defaults to None.
    """

    # logger.debug("Converting RTStruct: {0}".format(dcm_rt_file))
    # logger.debug("Using image series: {0}".format(dcm_img_list))
    # logger.debug("Output file prefix: {0}".format(prefix))
    # logger.debug("Output directory: {0}".format(output_dir))

    prefix = prefix + "{0}"

    dicom_image = sitk.ReadImage(dcm_img_list)
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

        # logger.debug("Image series to be converted to: {0}".format(image_output_path))

    if spacing:

        if isinstance(spacing, str):
            spacing = [float(i) for i in spacing.split(",")]

        # logger.debug("Overriding image spacing with: {0}".format(spacing))

    struct_list, struct_name_sequence = transform_point_set_from_dicom_struct(
        dicom_image, dicom_struct, spacing
    )
    # logger.debug("Converted all structures. Writing output.")
    for struct_index, struct_image in enumerate(struct_list):
        out_name = f"{prefix.format(struct_name_sequence[struct_index])}.nii.gz"
        out_name = output_dir.joinpath(out_name)
        # logger.debug(f"Writing file to: {output_dir}")
        sitk.WriteImage(struct_image, str(out_name))

    if image_output_path is not None:
        sitk.WriteImage(dicom_image, str(image_output_path))
