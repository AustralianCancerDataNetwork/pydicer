import logging
from pathlib import Path
import json
import SimpleITK as sitk
import pandas as pd
import pydicom

from radiomics import firstorder, shape, glcm, glrlm, glszm, ngtdm, gldm, imageoperations

from pydicer.utils import find_linked_image

logger = logging.getLogger(__name__)

PYRAD_DEFAULT_SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": "sitkNearestNeighbor",
    "verbose": True,
    "removeOutliers": 10000,
}

AVAILABLE_RADIOMICS = {
    "firstorder": firstorder.RadiomicsFirstOrder,
    "shape": shape.RadiomicsShape,
    "glcm": glcm.RadiomicsGLCM,
    "glrlm": glrlm.RadiomicsGLRLM,
    "glszm": glszm.RadiomicsGLSZM,
    "ngtdm": ngtdm.RadiomicsNGTDM,
    "gldm": gldm.RadiomicsGLDM,
}

FIRST_ORDER_FEATURES = firstorder.RadiomicsFirstOrder.getFeatureNames()
DEFAULT_RADIOMICS = {
    "firstorder": [f for f in FIRST_ORDER_FEATURES if not FIRST_ORDER_FEATURES[f]]
}


class AnalyseData:
    """
    Class that performs common analysis on converted data

    Args:
        data_directory (str|pathlib.Path, optional): Directory in which data is stored. Defaults to
            ".".
        dataset (str, optional): The name of the dataset on which to run analysis. Defaults to
            "nifti".
    """

    def __init__(self, data_directory=".", dataset_name="nifti"):
        self.data_directory = Path(data_directory)
        self.dataset_name = dataset_name
        self.dataset_directory = self.data_directory.joinpath(dataset_name)

    def get_all_computed_radiomics_for_dataset(self):
        """Return a DataFrame of radiomics computed for this dataset

        Returns:
            pd.DataFrame: The DataFrame of all radiomics computed for dataset
        """

        dfs = []
        for radiomics_file in self.dataset_directory.glob("**/structures/*.csv"):
            dfs.append(pd.read_csv(radiomics_file, index_col=0))

        df = pd.concat(dfs)
        df.sort_values(["Patient", "Contour"], inplace=True)

        return df

    def compute_radiomics(
        self,
        patient=None,
        radiomics=None,
        settings=None,
        structure_meta_data=None,
        image_meta_data=None,
        resample_to_image=False,
    ):
        """
        Visualise the data in the working directory. PNG files are generates providing a
        snapshot of the various data objects.

        Args:
            patient (list|str, optional): A patient ID (or list of patient IDs) to visualise.
            Defaults to None.
            radiomics (dict, optional): A dictionary of the pyradiomics to compute. Format should
                have the radiomic class name as the key and a list of feature names in the value.
                See https://pyradiomics.readthedocs.io/en/latest/features.html for more
                information.
                Defaults to all first order features.
            settings (dict, optional): Settings to pass to pyradiomics. Defaults to
                PYRAD_DEFAULT_SETTINGS.
            structure_meta_data (list, optional): A list of DICOM tags which will be extracted from
                the structure DICOM headers and included in the resulting table of radiomics.
                Defaults to None.
            image_meta_data ([type], optional): A list of DICOM tags which will be extracted from
                the image DICOM headers and included in the resulting table of radiomics.
                Defaults to None.
            resample_to_image (bool, optional): Define if mask should be resampled to image. If not
                the image will be resampled to mask. Defaults to False.

        Raises:
            ValueError: Raised if patient is not None, a list of strings or a string.
        """
        if isinstance(patient, list):
            if not all(isinstance(x, str) for x in patient):
                raise ValueError("All patient IDs must be of type 'str'")
        elif patient is None:
            patient = [p.name for p in self.dataset_directory.glob("*") if p.is_dir()]
        else:

            if not isinstance(patient, str) and patient is not None:
                raise ValueError(
                    "Patient ID must be list or str. None is valid to process all patients"
                )
            patient = [patient]

        if radiomics is None:
            radiomics = DEFAULT_RADIOMICS

        if settings is None:
            settings = PYRAD_DEFAULT_SETTINGS

        if structure_meta_data is None:
            structure_meta_data = []

        if image_meta_data is None:
            image_meta_data = []

        meta_data_cols = []

        for pat in patient:

            pat_dir_match = pat
            if pat_dir_match is None:
                pat_dir_match = "**"

            for struct_json in self.dataset_directory.glob(f"{pat_dir_match}/structures/*.json"):
                struct_dir = struct_json.parent.joinpath(struct_json.name.replace(".json", ""))

                img_file = find_linked_image(struct_dir)

                if img_file is None:
                    logger.error("Linked image %s not found")
                    continue
                img_json = img_file.parent.joinpath(img_file.name.replace(".nii.gz", ".json"))

                with open(img_json, "r", encoding="utf8") as json_file:
                    ds_dict = json.load(json_file)
                img_meta_data = pydicom.Dataset.from_json(
                    ds_dict, bulk_data_uri_handler=lambda _: None
                )

                with open(struct_json, "r", encoding="utf8") as json_file:
                    ds_dict = json.load(json_file)
                meta_data = pydicom.Dataset.from_json(
                    ds_dict, bulk_data_uri_handler=lambda _: None
                )

                output_frame = pd.DataFrame()
                for struct_nii in struct_dir.glob("*.nii.gz"):

                    struct_name = struct_nii.name.replace(".nii.gz", "")

                    # Reload the image for each new contour in case resampling is occuring,
                    # should start fresh each time.
                    image = sitk.ReadImage(str(img_file))
                    mask = sitk.ReadImage(str(struct_nii))

                    interpolator = settings.get("interpolator")
                    resample_pixel_spacing = settings.get("resampledPixelSpacing")

                    resample_pixel_spacing = list(image.GetSpacing())
                    settings["resampledPixelSpacing"] = resample_pixel_spacing

                    if resample_to_image:
                        resample_pixel_spacing = list(image.GetSpacing())
                        settings["resampledPixelSpacing"] = resample_pixel_spacing

                    if interpolator is not None and resample_pixel_spacing is not None:
                        image, mask = imageoperations.resampleImage(image, mask, **settings)

                    df_contour = pd.DataFrame()

                    for rad in radiomics:

                        if rad not in AVAILABLE_RADIOMICS:
                            logger.warning("Radiomic Class not found: %s", rad)
                            continue

                        radiomics_obj = AVAILABLE_RADIOMICS[rad]

                        features = radiomics_obj(image, mask, **settings)

                        features.disableAllFeatures()

                        # All features seem to be computed if all are disabled (possible
                        # pyradiomics bug?). Skip if all features in a class are disabled.
                        if len(radiomics[rad]) == 0:
                            continue

                        for feature in radiomics[rad]:
                            try:
                                features.enableFeatureByName(feature, True)
                            except LookupError:
                                # Feature not available in this set
                                logger.warning("Feature not found: %s", feature)

                        feature_result = features.execute()
                        feature_result = dict(
                            (f"{rad}|{key}", value) for (key, value) in feature_result.items()
                        )
                        df_feature_result = pd.DataFrame(feature_result, index=[struct_name])

                        # Merge the results
                        df_contour = pd.concat([df_contour, df_feature_result], axis=1)

                    output_frame = pd.concat([output_frame, df_contour])

                    # Add the meta data for this contour if there is any
                    for key in structure_meta_data:

                        col_key = f"struct|{key}"

                        if key in img_meta_data:
                            output_frame[col_key] = meta_data[key].value
                        else:
                            output_frame[col_key] = None

                        if col_key not in meta_data_cols:
                            meta_data_cols.append(col_key)

                # Add Image Series Data Object's Meta Data to the table
                for key in image_meta_data:

                    col_key = f"img|{key}"

                    value = None
                    if key in img_meta_data:
                        value = img_meta_data[key].value

                    output_frame[col_key] = pd.Series(
                        [value for p in range(len(output_frame.index))],
                        index=output_frame.index,
                    )

                    if col_key not in meta_data_cols:
                        meta_data_cols.append(col_key)

                output_frame.insert(loc=0, column="Patient", value=pat)
                output_frame.reset_index(inplace=True)
                columns = list(output_frame.columns)
                columns[0] = "Contour"
                output_frame.columns = columns

                struct_radiomics_filename = struct_json.name.replace(".json", ".csv")
                struct_radiomics_path = struct_json.parent.joinpath(struct_radiomics_filename)
                output_frame.to_csv(struct_radiomics_path)
