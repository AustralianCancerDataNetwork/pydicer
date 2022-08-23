import logging
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
from platipy.imaging import ImageVisualiser
from pydicer.constants import CONVERTED_DIR_NAME

from pydicer.utils import parse_patient_kwarg, load_object_metadata, read_converted_data

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
        self.output_directory = self.working_directory.joinpath(CONVERTED_DIR_NAME)

    def visualise(self, dataset_name=CONVERTED_DIR_NAME, patient=None, force=True):
        """Visualise the data in the working directory. PNG files are generates providing a
        snapshot of the various data objects.

        Args:
            dataset_name (str, optional): The name of the dataset to compute DVHs on. Defaults to
              "data" (runs on all data).
            patient (list|str, optional): A patient ID (or list of patient IDs) to visualise.
              Defaults to None.
            force (bool, optional): When True objects will be visualised even if the output files
              already exist. Defaults to True.
        """

        patient = parse_patient_kwarg(patient)
        df_process = read_converted_data(
            self.working_directory,
            dataset_name=dataset_name,
            patients=patient,
            join_working_directory=True,
        )

        for _, row in df_process[df_process["modality"] == "CT"].iterrows():
            img_path = Path(row.path)
            vis_filename = img_path.joinpath("CT.png")

            if vis_filename.exists() and not force:
                logger.info("Visualisation already exists at %s", vis_filename)
                continue

            img = sitk.ReadImage(str(img_path.joinpath(f"{row.modality}.nii.gz")))

            vis = ImageVisualiser(img)
            fig = vis.show()
            # load meta data from json file
            ds_dict = load_object_metadata(row)
            # deal with missing value in study description
            if "StudyDescription" not in ds_dict:
                ds_dict.StudyDescription = "NaN"
            # choose axis one
            # (this is the top-right box that is blank)
            ax = fig.axes[1]

            # choose a sensible font size
            # this will depend on the figure size you set
            fs = 9

            # insert metadata information
            ax.text(
                x=0.02,
                y=0.90,
                s=f"Patient ID: {ds_dict.PatientID}\n"
                f"Series Instance UID: \n"
                f"{ds_dict.SeriesInstanceUID}\n"
                f"Study Description: {ds_dict.StudyDescription}\n"
                f"Study Date: {ds_dict.StudyDate}",
                color="black",
                ha="left",
                va="top",
                size=fs,
                wrap=True,
                bbox=dict(boxstyle="square", fc="w", ec="r"),
            )

            fig.savefig(
                vis_filename,
                dpi=fig.dpi,
            )
            plt.close(fig)

            logger.debug("Created visualisation: %s", vis_filename)

        # Next visualise the structures on top of their linked image
        for _, row in df_process[df_process["modality"] == "RTSTRUCT"].iterrows():

            struct_dir = Path(row.path)

            # Find the linked image
            # TODO also render on images linked by Frame of Reference
            df_linked_img = df_process[
                df_process["sop_instance_uid"] == row.referenced_sop_instance_uid
            ]

            if len(df_linked_img) == 0:
                logger.warning(
                    "No linked images found, structures won't be visualised: %s",
                    row.sop_instance_uid,
                )

            for _, img_row in df_linked_img.iterrows():

                img_path = Path(img_row.path)

                # save image inside structure directory
                vis_filename = struct_dir.joinpath(f"vis_{img_row.hashed_uid}.png")

                if vis_filename.exists() and not force:
                    logger.info("Visualisation already exists at %s", vis_filename)
                    continue

                img = sitk.ReadImage(str(img_path.joinpath(f"{img_row.modality}.nii.gz")))

                vis = ImageVisualiser(img)
                masks = {
                    f.name.replace(".nii.gz", ""): sitk.ReadImage(str(f))
                    for f in struct_dir.glob("*.nii.gz")
                }

                if len(masks) == 0:
                    logger.warning("No contours found in structure directory: %s", {struct_dir})
                    continue

                vis.add_contour(masks)
                fig = vis.show()

                fig.savefig(vis_filename, dpi=fig.dpi)
                plt.close(fig)

                logger.debug("Created visualisation: %s", vis_filename)

        # Next visualise the doses on top of their linked image
        for _, row in df_process[df_process["modality"] == "RTDOSE"].iterrows():

            ## Currently doses are linked via: plan -> struct -> image

            # Find the linked plan
            df_linked_plan = df_process[
                df_process["sop_instance_uid"] == row.referenced_sop_instance_uid
            ]

            if len(df_linked_plan) == 0:
                logger.warning(
                    "No linked plans found, dose won't be visualised: %s", row.sop_instance_uid
                )
                continue

            # Find the linked structure set
            plan_row = df_linked_plan.iloc[0]
            df_linked_struct = df_process[
                df_process["sop_instance_uid"] == plan_row.referenced_sop_instance_uid
            ]

            if len(df_linked_struct) == 0:
                # Try to link via Frame of Reference instead
                df_linked_struct = df_process[
                    (df_process["modality"] == "RTSTRUCT") & (df_process["for_uid"] == row.for_uid)
                ]

            if len(df_linked_struct) == 0:
                logger.warning(
                    "No structures found, dose won't be visualised: %s", row.sop_instance_uid
                )
                continue

            # Find the linked image
            struct_row = df_linked_struct.iloc[0]
            df_linked_img = df_process[
                df_process["sop_instance_uid"] == struct_row.referenced_sop_instance_uid
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

                # save image inside dose directory
                vis_filename = dose_path.joinpath(f"vis_{struct_row.hashed_uid}.png")

                if vis_filename.exists() and not force:
                    logger.info("Visualisation already exists at %s", vis_filename)
                    continue

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
                    logger.warning("No contours found in structure directory: %s", {struct_dir})
                    continue

                vis.add_contour(masks)
                fig = vis.show()

                fig.savefig(vis_filename, dpi=fig.dpi)
                plt.close(fig)

                logger.debug("Created visualisation: %s", vis_filename)
