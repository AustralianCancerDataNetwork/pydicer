import logging
from pathlib import Path
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
from platipy.imaging import ImageVisualiser

from pydicer.utils import parse_patient_kwarg

logger = logging.getLogger(__name__)


class VisualiseData:
    """
    Class that facilitates the visualisation of the data once converted

    Args:
        working_directory (str|pathlib.Path, optional): Main working directory for pydicer.
            Defaults to ".".
    """

    def __init__(self, working_directory="."):
        self.working_directory = Path(working_directory)
        self.output_directory = self.working_directory.joinpath("data")

    def visualise(self, patient=None):
        """Visualise the data in the working directory. PNG files are generates providing a
        snapshot of the various data objects.

        Args:
            patient (list|str, optional): A patient ID (or list of patient IDs) to visualise.
            Defaults to None.
        """

        patient = parse_patient_kwarg(patient, self.output_directory)

        for pat in patient:

            # Read in the DataFrame storing the converted data for this patient
            converted_csv = self.output_directory.joinpath(pat, "converted.csv")
            if not converted_csv.exists():
                logger.warning("Converted CSV doesn't exist for %s", pat)
                continue

            df_converted = pd.read_csv(converted_csv, index_col=0)

            # first stage: visualise each image individually
            for _, row in df_converted[df_converted["modality"] == "CT"].iterrows():

                img_path = Path(row.path)

                img = sitk.ReadImage(str(img_path.joinpath(f"{row.modality}.nii.gz")))

                vis = ImageVisualiser(img)
                fig = vis.show()

                # save image alongside nifti
                vis_filename = img_path.joinpath("CT.png")
                fig.savefig(
                    vis_filename,
                    dpi=fig.dpi,
                )
                plt.close(fig)

                logger.debug("created visualisation: %s", vis_filename)

            # Next visualise the structures on top of their linked image
            for _, row in df_converted[df_converted["modality"] == "RTSTRUCT"].iterrows():

                struct_dir = Path(row.path)

                # Find the linked image
                # TODO also render on images linked by Frame of Reference
                df_linked_img = df_converted[
                    df_converted["sop_instance_uid"] == row.referenced_sop_instance_uid
                ]

                if len(df_linked_img) == 0:
                    logger.warning(
                        "No linked images found, structures won't be visualised: %s",
                        row.sop_instance_uid,
                    )

                for _, img_row in df_linked_img.iterrows():

                    img_path = Path(img_row.path)

                    img = sitk.ReadImage(str(img_path.joinpath(f"{img_row.modality}.nii.gz")))

                    vis = ImageVisualiser(img)
                    masks = {
                        f.name.replace(".nii.gz", ""): sitk.ReadImage(str(f))
                        for f in struct_dir.glob("*.nii.gz")
                    }

                    if len(masks) == 0:
                        logger.warning(
                            "No contours found in structure directory: %s", {struct_dir}
                        )
                        continue

                    vis.add_contour(masks)
                    fig = vis.show()

                    # save image inside structure directory
                    vis_filename = struct_dir.joinpath(f"vis_{img_row.hashed_uid}.png")
                    fig.savefig(vis_filename, dpi=fig.dpi)
                    plt.close(fig)

                    logger.debug("created visualisation: %s", vis_filename)

            # Next visualise the doses on top of their linked image
            for _, row in df_converted[df_converted["modality"] == "RTDOSE"].iterrows():

                ## Currently doses are linked via: plan -> struct -> image

                # Find the linked plan
                df_linked_plan = df_converted[
                    df_converted["sop_instance_uid"] == row.referenced_sop_instance_uid
                ]

                if len(df_linked_plan) == 0:
                    logger.warning(
                        "No linked plans found, dose won't be visualised: %s", row.sop_instance_uid
                    )
                    continue

                # Find the linked structure set
                plan_row = df_linked_plan.iloc[0]
                df_linked_struct = df_converted[
                    df_converted["sop_instance_uid"] == plan_row.referenced_sop_instance_uid
                ]

                if len(df_linked_struct) == 0:
                    # Try to link via Frame of Reference instead
                    df_linked_struct = df_converted[
                        (df_converted["modality"] == "RTSTRUCT")
                        & (df_converted["for_uid"] == row.for_uid)
                    ]

                if len(df_linked_struct) == 0:
                    logger.warning(
                        "No structures found, dose won't be visualised: %s", row.sop_instance_uid
                    )
                    continue

                # Find the linked image
                struct_row = df_linked_struct.iloc[0]
                df_linked_img = df_converted[
                    df_converted["sop_instance_uid"] == struct_row.referenced_sop_instance_uid
                ]

                if len(df_linked_img) == 0:
                    logger.warning(
                        "No linked images found, dose won't be visualised: %s",
                        row.sop_instance_uid,
                    )

                dose_path = Path(row.path)
                struct_dir = Path(struct_row.path)

                for _, img_row in df_linked_img.iterrows():

                    img_path = Path(img_row.path)

                    img = sitk.ReadImage(str(img_path.joinpath(f"{img_row.modality}.nii.gz")))
                    dose_img = sitk.ReadImage(str(dose_path.joinpath("RTDOSE.nii.gz")))
                    dose_img = sitk.Resample(dose_img, img)

                    vis = ImageVisualiser(img)

                    vis.add_scalar_overlay(
                        dose_img, "Dose", discrete_levels=20, colormap=plt.cm.get_cmap("inferno")
                    )

                    masks = {
                        f.name.replace(".nii.gz", ""): sitk.ReadImage(str(f))
                        for f in struct_dir.glob("*.nii.gz")
                    }

                    if len(masks) == 0:
                        logger.warning(
                            "No contours found in structure directory: %s", {struct_dir}
                        )
                        continue

                    vis.add_contour(masks)
                    fig = vis.show()

                    # save image inside structure directory
                    vis_filename = dose_path.joinpath(f"vis_{struct_row.hashed_uid}.png")
                    fig.savefig(vis_filename, dpi=fig.dpi)
                    plt.close(fig)

                    logger.debug("created visualisation: %s", vis_filename)
