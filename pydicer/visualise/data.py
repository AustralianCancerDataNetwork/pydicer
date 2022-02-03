import logging
from pathlib import Path
import SimpleITK as sitk
import matplotlib.pyplot as plt
from platipy.imaging import ImageVisualiser

logger = logging.getLogger(__name__)


class VisualiseData:
    """
    Class that facilitates the visualisation of the data once converted

    Args:
        output_directory (str|pathlib.Path, optional): Directory in which converted data is stored.
            Defaults to ".".
    """

    def __init__(self, output_directory="."):
        self.output_directory = Path(output_directory)

    def visualise(self):
        """
        Function to visualise the data
        """

        # first stage: visualise each image individually
        for img_filename in self.output_directory.glob("**/images/*.nii.gz"):

            img = sitk.ReadImage(str(img_filename))

            vis = ImageVisualiser(img)
            fig = vis.show()

            # save image alongside nifti
            vis_filename = img_filename.parent / img_filename.name.replace(
                "".join(img_filename.suffixes), ".png"
            )
            fig.savefig(
                vis_filename,
                dpi=fig.dpi,
            )
            plt.close(fig)

            logger.debug("created visualisation: %s", vis_filename)

        # Next visualise the structures on top of their linked image
        for struct_dir in self.output_directory.glob("**/structures/*"):

            # Make sure this is a structure directory
            if not struct_dir.is_dir():
                continue

            img_id = struct_dir.name.split("_")[1]

            img_links = list(struct_dir.parent.parent.glob(f"images/*{img_id}.nii.gz"))

            # If we have multiple linked images (not sure if this can happen but it might?) then
            # take the first one. If we find no linked images log and error and don't visualise for
            # now
            if len(img_links) == 0:
                logger.error("Linked image %s not found", img_id)
                continue

            img_file = img_links[0]

            img = sitk.ReadImage(str(img_file))

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

            # save image inside structure directory
            vis_filename = struct_dir.parent.joinpath(f"{struct_dir.name}.png")
            fig.savefig(
                vis_filename,
                dpi=fig.dpi,
            )
            plt.close(fig)

            logger.debug("created visualisation: %s", vis_filename)

        # Next visualise the doses on top of their linked image
        for dose_file in self.output_directory.glob("**/doses/*.nii.gz"):

            ## Currently doses are linked via: plan -> struct -> image

            # Find the linked plan
            link_id = dose_file.name.replace(".nii.gz", "").split("_")[-1]
            plan_links = list(dose_file.parent.parent.glob(f"plans/*{link_id}_*.json"))

            if len(plan_links) == 0:
                logger.error("Linked plan %s not found", link_id)

                # Check for image link
                img_links = list(dose_file.parent.parent.glob(f"images/*{link_id}.nii.gz"))

            struct_dir = None
            if len(plan_links) > 0:
                # Find the linked struct
                struct_id = plan_links[0].name.replace(".json", "").split("_")[-1]
                struct_links = [
                    p for p in dose_file.parent.parent.glob(f"structures/{struct_id}_*/") if p.is_dir()
                ]

                if len(struct_links) == 0:
                    logger.error("Linked structure %s not found", struct_id)
                    continue

                # Finally find the linked image
                struct_dir = struct_links[0]
                img_id = struct_dir.name.split("_")[1]
                img_links = list(struct_dir.parent.parent.glob(f"images/*{img_id}.nii.gz"))

            if len(img_links) == 0:
                logger.error("Linked image %s not found", img_id)
                continue

            img_file = img_links[0]

            img = sitk.ReadImage(str(img_file))
            dose_img = sitk.ReadImage(str(dose_file))
            dose_img = sitk.Resample(dose_img, img)

            vis = ImageVisualiser(img)

            vis.add_scalar_overlay(
                dose_img, "Dose", discrete_levels=20, colormap=plt.cm.get_cmap("inferno")
            )

            if struct_dir is not None:
                masks = {
                    f.name.replace(".nii.gz", ""): sitk.ReadImage(str(f))
                    for f in struct_dir.glob("*.nii.gz")
                }

                if len(masks) > 0:
                    vis.add_contour(masks, linewidth=1)

            fig = vis.show()

            # save image inside doses directory
            file_name = dose_file.name.replace(".nii.gz", ".png")
            vis_filename = dose_file.parent.joinpath(f"{file_name}")
            fig.savefig(
                vis_filename,
                dpi=fig.dpi,
            )
            plt.close(fig)

            logger.debug("created visualisation: %s", vis_filename)
