import logging
from pathlib import Path
import pydicom 
import json
import SimpleITK as sitk
import numpy as np 
import matplotlib.pyplot as plt
import textwrap
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

    def visualise(self, patient=None):
        """Visualise the data in the working directory. PNG files are generates providing a
        snapshot of the various data objects.

        Args:
            patient (list|str, optional): A patient ID (or list of patient IDs) to visualise.
            Defaults to None.
        """

        if isinstance(patient, list):
            if not all(isinstance(x, str) for x in patient):
                raise ValueError("All patient IDs must be of type 'str'")
        else:

            if not isinstance(patient, str) and patient is not None:
                raise ValueError(
                    "Patient ID must be list or str. None is a valid to process all patients"
                )
            patient = [patient]

        for pat in patient:

            pat_dir_match = pat
            if pat_dir_match is None:
                pat_dir_match = "**"

            # first stage: visualise each image individually
            for img_filename in self.output_directory.glob(f"{pat_dir_match}/images/*.nii.gz"):
                img = sitk.ReadImage(str(img_filename))

                vis = ImageVisualiser(img)
                fig = vis.show()
                
                # load meta data from json file 
                img_json = img_filename.parent.joinpath(img_filename.name.replace(".nii.gz", ".json"))

                with open(img_json, "r", encoding="utf8") as json_file:
                    ds_dict = json.load(json_file)

                # load the metadata back into a pydicom dataset
                img_meta_data = pydicom.Dataset.from_json(
                    ds_dict, bulk_data_uri_handler=lambda _: None
                )
            
                # print all attributes in pydicom dataset 
                # print(img_meta_data)
                # break
                
                # print study description, will cause error messages in some cases
                # print("study description: ")
                # print(img_meta_data.StudyDescription)
                
                # choose axis one
                # (this is the top-right box that is blank)
                ax = fig.axes[1]

                # choose a sensible font size
                # this will depend on the figure size you set
                fs = 9
                
                # insert metadata information
                txt = ax.text(
                    x=0.03,
                    y=0.90,
                    s= "Patient ID: " + img_meta_data.PatientID + \
                    "\nSeries Instance UID: \n" + img_meta_data.SeriesInstanceUID +
                    "\nStudy Instance UID: \n" + img_meta_data.StudyInstanceUID + 
                    "\nStudy Date: " + img_meta_data.StudyDate,
                    color="black",
                    ha="left",
                    va="top",
                    size=fs,
                    wrap=True,
                    bbox=dict(boxstyle='square', fc='w', ec='r')
                )
                
                txt._get_wrap_line_width = lambda : 300.
                
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
            for struct_dir in self.output_directory.glob(f"{pat_dir_match}/structures/*"):

                # Make sure this is a structure directory
                if not struct_dir.is_dir():
                    continue

                img_id = struct_dir.name.split("_")[1]

                img_links = list(struct_dir.parent.parent.glob(f"images/*{img_id}.nii.gz"))

                # If we have multiple linked images (not sure if this can happen but it might?)
                # then take the first one. If we find no linked images log and error and don't
                # visualise for now
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
            for dose_file in self.output_directory.glob(f"{pat_dir_match}/doses/*.nii.gz"):

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
                        p
                        for p in dose_file.parent.parent.glob(f"structures/{struct_id}_*/")
                        if p.is_dir()
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


if __name__ == "__main__":
    visualise_data = VisualiseData()
    visualise_data.visualise()
