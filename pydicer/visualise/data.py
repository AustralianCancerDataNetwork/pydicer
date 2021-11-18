import logging
from pathlib import Path
import SimpleITK as sitk
from platipy.imaging import ImageVisualiser

logger = logging.getLogger(__name__)


class VisualiseData:
    """
    Class that facilitates the visualisation of the data once converted

    Args:
        - :
        df_preprocess (pd.DataFrame): the DataFrame which was produced by PreprocessData
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

            logger.debug("created visualisation%s", vis_filename)
