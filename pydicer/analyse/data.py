import logging
from pathlib import Path
import re

import SimpleITK as sitk
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from radiomics import firstorder, shape, glcm, glrlm, glszm, ngtdm, gldm, imageoperations
from platipy.imaging.dose.dvh import calculate_dvh_for_labels

from pydicer.utils import load_object_metadata

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

    def __init__(self, data_directory=".", dataset_name="data"):
        self.working_directory = Path(data_directory)
        self.dataset_directory = self.working_directory.joinpath(dataset_name)

    def get_all_computed_radiomics_for_dataset(self):
        """Return a DataFrame of radiomics computed for this dataset

        Returns:
            pd.DataFrame: The DataFrame of all radiomics computed for dataset
        """

        dfs = []
        for struct_dir in self.dataset_directory.glob("*/structures/*"):
            for radiomics_file in struct_dir.glob("radiomics_*.csv"):
                col_types = {
                    "Contour": str,
                    "Patient": str,
                    "ImageHashedUID": str,
                    "StructHashedUID": str,
                }
                df_rad = pd.read_csv(radiomics_file, index_col=0, dtype=col_types)
                dfs.append(df_rad)

        df = pd.concat(dfs)
        df.sort_values(["Patient", "ImageHashedUID", "StructHashedUID", "Contour"], inplace=True)

        return df

    def compute_radiomics(
        self,
        patient=None,
        radiomics=None,
        settings=None,
        structure_match_regex=None,
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
            structure_match_regex (str, optional): Regular expression to select structures to
                compute radiomics for. Defaults to None.
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
                    "Patient ID must be list or str. None is a valid to process all patients"
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

            # Read in the DataFrame storing the converted data for this patient
            converted_csv = self.dataset_directory.joinpath(pat, "converted.csv")
            if not converted_csv.exists():
                logger.warning("Converted CSV doesn't exist for %s", pat)
                continue

            df_converted = pd.read_csv(converted_csv, index_col=0)

            # Next visualise the structures on top of their linked image
            for _, struct_row in df_converted[df_converted["modality"] == "RTSTRUCT"].iterrows():

                struct_dir = Path(struct_row.path)

                # Find the linked image
                # TODO also render on images linked by Frame of Reference
                df_linked_img = df_converted[
                    df_converted["sop_instance_uid"] == struct_row.referenced_sop_instance_uid
                ]

                if len(df_linked_img) == 0:
                    logger.warning(
                        "No linked images found, structures won't be visualised: %s",
                        struct_row.sop_instance_uid,
                    )

                for _, img_row in df_linked_img.iterrows():

                    img_file = Path(img_row.path).joinpath(f"{img_row.modality}.nii.gz")
                    img_meta_data = load_object_metadata(img_row)

                    struct_meta_data = load_object_metadata(struct_row)

                    output_frame = pd.DataFrame()
                    for struct_nii in struct_dir.glob("*.nii.gz"):

                        struct_name = struct_nii.name.replace(".nii.gz", "")

                        # If a regex is set, make sure this structure name matches it
                        if structure_match_regex:
                            if re.search(structure_match_regex, struct_name) is None:
                                continue

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

                            if key in struct_meta_data:
                                output_frame[col_key] = struct_meta_data[key].value
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

                    output_frame.insert(
                        loc=0, column="StructHashedUID", value=struct_row.hashed_uid
                    )
                    output_frame.insert(loc=0, column="ImageHashedUID", value=img_row.hashed_uid)
                    output_frame.insert(loc=0, column="Patient", value=pat)
                    output_frame.reset_index(inplace=True)
                    columns = list(output_frame.columns)
                    columns[0] = "Contour"
                    output_frame.columns = columns

                    struct_radiomics_path = struct_dir.joinpath(
                        f"radiomics_{img_row.hashed_uid}.csv"
                    )
                    output_frame.to_csv(struct_radiomics_path)

    def compute_dvh(
        self, patient=None, bin_width=0.1, structure_meta_data_cols=None, dose_meta_data_cols=None
    ):
        """
        Compute the Dose Volume Histogram (DVH) for dose volumes and linked structures.

        Args:
            patient (list|str, optional): A patient ID (or list of patient IDs) to compute DVH for.
            Defaults to None.
            bin_width (float, optional): The bin width of the Dose Volume Histogram.
            structure_meta_data (list, optional): A list of DICOM tags which will be extracted from
                the structure DICOM headers and included in the resulting table of radiomics.
                Defaults to None.
            structure_meta_data (list, optional): A list of DICOM tags which will be extracted from
                the Dose DICOM headers and included in the resulting table of radiomics.
                Defaults to None.

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
                    "Patient ID must be list or str. None is a valid to process all patients"
                )
            patient = [patient]

        if structure_meta_data_cols is None:
            structure_meta_data_cols = []

        if dose_meta_data_cols is None:
            dose_meta_data_cols = []

        meta_data_cols = []

        for pat in patient:

            # Read in the DataFrame storing the converted data for this patient
            converted_csv = self.dataset_directory.joinpath(pat, "converted.csv")
            if not converted_csv.exists():
                logger.warning("Converted CSV doesn't exist for %s", pat)
                continue

            df_converted = pd.read_csv(converted_csv, index_col=0)

            # For each dose, find the structures in the same frame of reference and compute the
            # DVH
            for _, dose_row in df_converted[df_converted["modality"] == "RTDOSE"].iterrows():

                ## Currently doses are linked via: plan -> struct -> image
                dose_meta_data = load_object_metadata(dose_row)

                # Find the linked plan
                df_linked_plan = df_converted[
                    df_converted["sop_instance_uid"] == dose_row.referenced_sop_instance_uid
                ]

                if len(df_linked_plan) == 0:
                    logger.warning("No linked plans found for dose: %s", dose_row.sop_instance_uid)

                # Find the linked structure set
                df_linked_struct = None
                if len(df_linked_plan) > 0:
                    plan_row = df_linked_plan.iloc[0]
                    df_linked_struct = df_converted[
                        df_converted["sop_instance_uid"] == plan_row.referenced_sop_instance_uid
                    ]

                # Also link via Frame of Reference
                df_for_linked = df_converted[
                    (df_converted["modality"] == "RTSTRUCT")
                    & (df_converted["for_uid"] == dose_row.for_uid)
                ]

                if df_linked_struct is None:
                    df_linked_struct = df_for_linked
                else:
                    df_linked_struct = pd.concat([df_linked_struct, df_for_linked])

                if len(df_linked_struct) == 0:
                    logger.warning("No structures found for plan: %s", plan_row.sop_instance_uid)

                dose_file = Path(dose_row.path).joinpath("RTDOSE.nii.gz")

                for _, struct_row in df_linked_struct.iterrows():

                    struct_hash = struct_row.hashed_uid
                    dvh_csv = dose_file.parent.joinpath(f"dvh_{struct_hash}.csv")

                    struct_dir = Path(struct_row.path)

                    struct_meta_data = load_object_metadata(struct_row)

                    dose = sitk.ReadImage(str(dose_file))

                    structures = {
                        struct_nii.name.replace(".nii.gz", ""): sitk.ReadImage(str(struct_nii))
                        for struct_nii in struct_dir.glob("*.nii.gz")
                    }
                    if len(structures) == 0:
                        continue

                    dvh = calculate_dvh_for_labels(dose, structures, bin_width=bin_width)

                    # Save the DVH plot
                    dvh_file = dose_file.parent.joinpath(f"dvh_{struct_hash}.png")
                    plt_dvh = dvh.melt(
                        id_vars=["label", "cc", "mean"], var_name="bin", value_name="dose"
                    )

                    sns.set(rc={"figure.figsize": (16.7, 12.27)})
                    p = sns.lineplot(data=plt_dvh, x="bin", y="dose", hue="label", palette="Dark2")
                    p.set(
                        xlabel="Dose (Gy)", ylabel="Frequency", title="Dose Volume Histogram (DVH)"
                    )
                    p.get_figure().savefig(dvh_file)
                    plt.close(p.get_figure())

                    # Add Dose Data Object's Meta Data to the table
                    for key in dose_meta_data_cols:

                        col_key = f"dose|{key}"

                        value = None
                        if key in dose_meta_data:
                            value = dose_meta_data[key].value

                        dvh[col_key] = pd.Series(
                            [value for p in range(len(dvh.index))],
                            index=dvh.index,
                        )

                        if col_key not in meta_data_cols:
                            meta_data_cols.append(col_key)

                    # Add Structure Data Object's Meta Data to the table
                    for key in structure_meta_data_cols:

                        col_key = f"struct|{key}"

                        value = None
                        if key in struct_meta_data:
                            value = struct_meta_data[key].value

                        dvh[col_key] = pd.Series(
                            [value for p in range(len(dvh.index))],
                            index=dvh.index,
                        )

                        if col_key not in meta_data_cols:
                            meta_data_cols.append(col_key)

                    dvh.insert(loc=0, column="struct_hash", value=struct_hash)
                    dvh.insert(loc=0, column="patient", value=pat)

                    # Save DVH CSV
                    dvh.to_csv(dvh_csv)
