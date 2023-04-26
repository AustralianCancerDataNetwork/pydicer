import logging
from pathlib import Path
import re

import SimpleITK as sitk
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from platipy.imaging.dose.dvh import (
    calculate_dvh_for_labels,
    calculate_d_x,
    calculate_v_x,
    calculate_d_cc_x,
)
from pydicer.constants import CONVERTED_DIR_NAME

from pydicer.utils import (
    load_object_metadata,
    load_dvh,
    parse_patient_kwarg,
    read_converted_data,
    get_iterator,
    get_structures_linked_to_dose,
)
from pydicer.logger import PatientLogger

logger = logging.getLogger(__name__)


PYRAD_DEFAULT_SETTINGS = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": "sitkNearestNeighbor",
    "verbose": True,
    "removeOutliers": 10000,
}


class AnalyseData:
    """
    Class that performs common analysis on converted data

    Args:
        working_directory (str|pathlib.Path, optional): Directory in which data is stored. Defaults
          to ".".
    """

    def __init__(self, working_directory="."):
        self.working_directory = Path(working_directory)
        self.output_directory = self.working_directory.joinpath(CONVERTED_DIR_NAME)

    def get_all_computed_radiomics_for_dataset(
        self, dataset_name=CONVERTED_DIR_NAME, patient=None
    ):
        """Return a DataFrame of radiomics computed for this dataset

        Args:
            dataset_name (str, optional): The name of the dataset on which to run analysis.
              Defaults to "data".
            patient (list|str, optional): A patient ID (or list of patient IDs) to fetch radiomics
              for. Defaults to None.

        Returns:
            pd.DataFrame: The DataFrame of all radiomics computed for dataset
        """

        patient = parse_patient_kwarg(patient)

        df_data = read_converted_data(self.working_directory, dataset_name, patients=patient)

        dfs = []
        for _, struct_row in df_data[df_data["modality"] == "RTSTRUCT"].iterrows():

            struct_dir = Path(struct_row.path)

            for radiomics_file in struct_dir.glob("radiomics_*.csv"):
                col_types = {
                    "Contour": str,
                    "Patient": str,
                    "ImageHashedUID": str,
                    "StructHashedUID": str,
                }
                df_rad = pd.read_csv(radiomics_file, index_col=0, dtype=col_types)
                dfs.append(df_rad)

        if len(dfs) == 0:
            return pd.DataFrame(
                columns=["Patient", "ImageHashedUID", "StructHashedUID", "Contour"]
            )

        df = pd.concat(dfs)
        df.sort_values(["Patient", "ImageHashedUID", "StructHashedUID", "Contour"], inplace=True)

        return df

    def get_all_dvhs_for_dataset(self, dataset_name=CONVERTED_DIR_NAME, patient=None):
        """Return a DataFrame of DVHs computed for this dataset

        Args:
            dataset_name (str, optional): The name of the dataset on which to run analysis.
              Defaults to "data".
            patient (list|str, optional): A patient ID (or list of patient IDs) to fetch DVHs for.
              Defaults to None.

        Returns:
            pd.DataFrame: The DataFrame of all DVHs computed for dataset
        """

        patient = parse_patient_kwarg(patient)

        df_data = read_converted_data(self.working_directory, dataset_name, patients=patient)

        df_result = pd.DataFrame(columns=["patient", "struct_hash", "dose_hash", "label"])
        for _, dose_row in df_data[df_data["modality"] == "RTDOSE"].iterrows():

            struct_hashes = df_data[
                (df_data.for_uid == dose_row.for_uid) & (df_data.modality == "RTSTRUCT")
            ].hashed_uid.tolist()

            df_result = pd.concat([df_result, load_dvh(dose_row, struct_hash=struct_hashes)])

        return df_result

    def compute_dose_metrics(
        self,
        dataset_name=CONVERTED_DIR_NAME,
        patient=None,
        df_process=None,
        d_point=None,
        v_point=None,
        d_cc_point=None,
    ):
        """Compute Dose metrics from a DVH

        Args:
            dataset_name (str, optional): The name of the dataset from which to extract dose
               metrics. Defaults to "data".
            patient (list|str, optional): A patient ID (or list of patient IDs) to compute
              dose metrics for. Must be None if df_process is provided. Defaults to None.
            df_process (pd.DataFrame, optional): A DataFrame of the objects to compute dose metrics
              for. Must be None if patient is provided. Defaults to None.
            d_point (float|int|list, optional): The point or list of points at which to compute the
              D metric. E.g. to compute D50, D95 and D99, supply [50, 95, 99]. Defaults to None.
            v_point (float|int|list, optional): The point or list of points at which to compute the
              V metric. E.g. to compute V5, V10 and V50, supply [5, 10, 50]. Defaults to None.
            d_cc_point (float|int|list, optional): The point or list of points at which to compute
              the Dcc metric. E.g. to compute Dcc5, Dcc10 and Dcc50, supply [5, 10, 50]. Defaults
              to None.

        Raises:
            ValueError: One of d_point, v_point or d_cc_point should be set
            ValueError: Points must be of type float or int

        Returns:
            pd.DataFrame: The DataFrame containing the requested metrics.
        """

        if patient is not None and df_process is not None:
            raise ValueError("Only one of patient and df_process pay be provided.")

        if df_process is None:
            patient = parse_patient_kwarg(patient)
            df_process = read_converted_data(
                self.working_directory, dataset_name=dataset_name, patients=patient
            )

        if d_point is None and v_point is None and d_cc_point is None:
            raise ValueError("One of d_point, v_point or d_cc_point should be set")

        if not isinstance(d_point, list):
            if d_point is None:
                d_point = []
            else:
                d_point = [d_point]

        if not all(isinstance(x, (int, float)) for x in d_point):
            raise ValueError("D point must be of type int or float")

        if not isinstance(v_point, list):
            if v_point is None:
                v_point = []
            else:
                v_point = [v_point]

        if not all(isinstance(x, (int, float)) for x in v_point):
            raise ValueError("V point must be of type int or float")

        if not isinstance(d_cc_point, list):
            if d_cc_point is None:
                d_cc_point = []
            else:
                d_cc_point = [d_cc_point]

        if not all(isinstance(x, (int, float)) for x in d_cc_point):
            raise ValueError("D_cc point must be of type int or float")

        df_result = pd.DataFrame()

        # For each dose, find the structures in the same frame of reference and compute the
        # DVH
        df_doses = df_process[df_process["modality"] == "RTDOSE"]
        for _, row in get_iterator(
            df_doses.iterrows(), length=len(df_doses), unit="objects", name="Compute Dose Metrics"
        ):
            patient_logger = PatientLogger(row.patient_id, self.output_directory, force=False)

            ## Currently doses are linked via: plan -> struct -> image

            # Find the linked plan
            df_linked_plan = df_process[
                df_process["sop_instance_uid"] == row.referenced_sop_instance_uid
            ]

            if len(df_linked_plan) == 0:
                logger.warning("No linked plans found for dose: %s", row.sop_instance_uid)

            # Find the linked structure set
            df_linked_struct = None
            if len(df_linked_plan) > 0:
                plan_row = df_linked_plan.iloc[0]
                df_linked_struct = df_process[
                    df_process["sop_instance_uid"] == plan_row.referenced_sop_instance_uid
                ]

            # Also link via Frame of Reference
            df_for_linked = df_process[
                (df_process["modality"] == "RTSTRUCT") & (df_process["for_uid"] == row.for_uid)
            ]

            struct_hashes = (
                df_linked_struct.hashed_uid.tolist() + df_for_linked.hashed_uid.tolist()
            )

            dvh = load_dvh(row, struct_hash=struct_hashes)

            if len(dvh) == 0:
                logger.warning("No DVHs found for %s", struct_hashes)
                continue

            df = dvh[["patient", "struct_hash", "dose_hash", "label", "cc", "mean"]]
            df_d = calculate_d_x(dvh, d_point)
            df_v = calculate_v_x(dvh, v_point)
            df_dcc = calculate_d_cc_x(dvh, d_cc_point)
            df = pd.concat([df, df_d, df_v, df_dcc], axis=1)
            df = df.loc[:, ~df.columns.duplicated()]

            df_result = pd.concat([df_result, df])

            patient_logger.eval_module_process("analyse_compute_dose_metrics", row.hashed_uid)

        return df_result

    def compute_radiomics(
        self,
        dataset_name=CONVERTED_DIR_NAME,
        patient=None,
        df_process=None,
        force=True,
        radiomics=None,
        settings=None,
        structure_match_regex=None,
        structure_meta_data=None,
        image_meta_data=None,
        resample_to_image=False,
    ):
        """
        Compute radiomics for the data in the working directory. Results are saved as csv files in
        the structure directories processed.

        Args:
            dataset_name (str, optional): The name of the dataset to compute radiomics on. Defaults
              to "data" (runs on all data).
            patient (list|str, optional): A patient ID (or list of patient IDs) to compute
              radiomics for. Must be None if df_process is provided. Defaults to None.
            df_process (pd.DataFrame, optional): A DataFrame of the objects to compute radiomics
              for. Must be None if patient is provided. Defaults to None.
            force (bool, optional): When True, radiomics will be recomputed even if the output file
              already exists. Defaults to True.
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
            image_meta_data (list, optional): A list of DICOM tags which will be extracted from
                the image DICOM headers and included in the resulting table of radiomics.
                Defaults to None.
            resample_to_image (bool, optional): Define if mask should be resampled to image. If not
                the image will be resampled to mask. Defaults to False.

        Raises:
            ValueError: Raised if patient is not None, a list of strings or a string.
        """

        # Begin pyradiomics workaround
        # This code should be moved back to the top of this file once pyradiomics integration into
        # poetry issue is resolved: https://github.com/AIM-Harvard/pyradiomics/issues/787

        try:
            # pylint: disable=import-outside-toplevel
            from radiomics import (
                firstorder,
                shape,
                glcm,
                glrlm,
                glszm,
                ngtdm,
                gldm,
                imageoperations,
            )
        except ImportError:
            print(
                "Due to some limitations in the current version of pyradiomics, pyradiomics "
                "must be installed separately. Please run `pip install pyradiomics` to use the "
                "compute radiomics functionality."
            )
            return

        # pylint: disable=invalid-name
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

        # End pyradiomics workaround

        if patient is not None and df_process is not None:
            raise ValueError("Only one of patient and df_process pay be provided.")

        if df_process is None:
            patient = parse_patient_kwarg(patient)
            df_process = read_converted_data(
                self.working_directory, dataset_name=dataset_name, patients=patient
            )

        # Read all converted data for linkage
        df_converted = read_converted_data(self.working_directory)

        if radiomics is None:
            radiomics = DEFAULT_RADIOMICS

        if settings is None:
            settings = PYRAD_DEFAULT_SETTINGS

        if structure_meta_data is None:
            structure_meta_data = []

        if image_meta_data is None:
            image_meta_data = []

        meta_data_cols = []

        # Next compute the radiomics for each structure using their linked image
        df_structs = df_process[df_process["modality"] == "RTSTRUCT"]

        for _, struct_row in get_iterator(
            df_structs.iterrows(), length=len(df_structs), unit="objects", name="Compute Radiomics"
        ):
            patient_logger = PatientLogger(
                struct_row.patient_id, self.output_directory, force=False
            )

            struct_dir = Path(struct_row.path)

            # Find the linked image
            df_linked_img = df_converted[
                (df_converted["sop_instance_uid"] == struct_row.referenced_sop_instance_uid)
                | (
                    (df_converted["for_uid"] == struct_row.for_uid)
                    & (df_converted["modality"].isin(["CT", "MR", "PT"]))
                )
            ]

            if len(df_linked_img) == 0:
                logger.warning(
                    "No linked images found, structures won't be visualised: %s",
                    struct_row.sop_instance_uid,
                )

            for _, img_row in df_linked_img.iterrows():

                struct_radiomics_path = struct_dir.joinpath(f"radiomics_{img_row.hashed_uid}.csv")

                if struct_radiomics_path.exists() and not force:
                    logger.info("Radiomics already computed at %s", struct_radiomics_path)
                    continue

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

                output_frame.insert(loc=0, column="StructHashedUID", value=struct_row.hashed_uid)
                output_frame.insert(loc=0, column="ImageHashedUID", value=img_row.hashed_uid)
                output_frame.insert(loc=0, column="Patient", value=struct_row.patient_id)
                output_frame.reset_index(inplace=True)
                columns = list(output_frame.columns)
                columns[0] = "Contour"
                output_frame.columns = columns

                output_frame.to_csv(struct_radiomics_path)
                patient_logger.eval_module_process(
                    "analyse_compute_radiomics", struct_row.hashed_uid
                )

    def compute_dvh(
        self,
        dataset_name=CONVERTED_DIR_NAME,
        patient=None,
        df_process=None,
        force=True,
        bin_width=0.1,
        structure_meta_data_cols=None,
        dose_meta_data_cols=None,
    ):
        """
        Compute the Dose Volume Histogram (DVH) for dose volumes and linked structures.

        Args:
            dataset_name (str, optional): The name of the dataset to compute DVHs on. Defaults to
              "data" (runs on all data).
            patient (list|str, optional): A patient ID (or list of patient IDs) to compute DVH
              for. Must be None if df_process is provided. Defaults to None.
            df_process (pd.DataFrame, optional): A DataFrame of the objects to compute radiomics
              for. Must be None if patient is provided. Defaults to None.
            force (bool, optional): When True, DVHs will be recomputed even if the output file
              already exists. Defaults to True.
            bin_width (float, optional): The bin width of the Dose Volume Histogram.
            structure_meta_data_cols (list, optional): A list of DICOM tags which will be extracted
                from the structure DICOM headers and included in the resulting table of radiomics.
                Defaults to None.
            dose_meta_data_cols (list, optional): A list of DICOM tags which will be extracted from
                the Dose DICOM headers and included in the resulting table of radiomics.
                Defaults to None.

        Raises:
            ValueError: Raised if patient is not None, a list of strings or a string.
        """

        if patient is not None and df_process is not None:
            raise ValueError("Only one of patient and df_process pay be provided.")

        if df_process is None:
            patient = parse_patient_kwarg(patient)
            df_process = read_converted_data(
                self.working_directory, dataset_name=dataset_name, patients=patient
            )

        if structure_meta_data_cols is None:
            structure_meta_data_cols = []

        if dose_meta_data_cols is None:
            dose_meta_data_cols = []

        meta_data_cols = []

        # For each dose, find the structures in the same frame of reference and compute the
        # DVH
        df_doses = df_process[df_process["modality"] == "RTDOSE"]
        for _, dose_row in get_iterator(
            df_doses.iterrows(), length=len(df_doses), unit="objects", name="Compute DVH"
        ):
            dose_meta_data = load_object_metadata(dose_row)

            df_linked_struct = get_structures_linked_to_dose(self.working_directory, dose_row)

            if len(df_linked_struct) == 0:
                logger.warning(
                    "No linked structures found for dose: %s", dose_row.sop_instance_uid
                )

            dose_file = Path(dose_row.path).joinpath("RTDOSE.nii.gz")

            for _, struct_row in df_linked_struct.iterrows():

                struct_hash = struct_row.hashed_uid
                dvh_csv = dose_file.parent.joinpath(f"dvh_{struct_hash}.csv")

                if dvh_csv.exists() and not force:
                    logger.info("DVH already computed at %s", dvh_csv)
                    continue

                logger.info(
                    "Computing DVH for dose %s on structures %s for patient %s",
                    dose_row.hashed_uid,
                    struct_row.hashed_uid,
                    struct_row.patient_id,
                )

                struct_dir = Path(struct_row.path)

                struct_meta_data = load_object_metadata(struct_row)

                dose = sitk.ReadImage(str(dose_file))

                structures = {
                    struct_nii.name.replace(".nii.gz", ""): sitk.ReadImage(str(struct_nii))
                    for struct_nii in struct_dir.glob("*.nii.gz")
                }
                if len(structures) == 0:
                    logger.debug("No structures found in: %s", struct_dir)
                    continue

                dvh = calculate_dvh_for_labels(dose, structures, bin_width=bin_width)

                # Save the DVH plot
                dvh_file = dose_file.parent.joinpath(f"dvh_{struct_hash}.png")
                plt_dvh = dvh.melt(
                    id_vars=["label", "cc", "mean"], var_name="bin", value_name="dose"
                )

                sns.set(rc={"figure.figsize": (16.7, 12.27)})
                p = sns.lineplot(data=plt_dvh, x="bin", y="dose", hue="label", palette="Dark2")
                p.set(xlabel="Dose (Gy)", ylabel="Frequency", title="Dose Volume Histogram (DVH)")
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

                dvh.insert(loc=0, column="dose_hash", value=dose_row.hashed_uid)
                dvh.insert(loc=0, column="struct_hash", value=struct_hash)
                dvh.insert(loc=0, column="patient", value=dose_row.patient_id)

                # Save DVH CSV
                dvh.to_csv(dvh_csv)
