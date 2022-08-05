import logging
from datetime import time, datetime
import numpy as np
import SimpleITK as sitk
import pydicom as pdcm

logger = logging.getLogger(__name__)


def convert_dicom_to_nifti_pt(
    input_filepaths,
    output_filepath,
    default_patient_weight=None,
):
    """Function to convert the dicom files contained in input_filepaths to one NIFTI image.

    Args:
        input_filepaths (list): list of the dicom paths
        output_filepath (str): path to the output file path where to store the NIFTI file.
        default_patient_weight (float, optional): If the patient's weight is missing from the PT
            DICOM it can be provided through this argument. Defaults to None.

    Raises:
        MissingWeightError: Error to alert when the weight is missing from the PT, to compute
            the SUV.
        RuntimeError: Error to alert when one or more slices are missing
        ValueError: Raised when a modality or a unit (for the PT) is not handled.

    Returns:
        numpy.array: The numpy image, used to compute the bounding boxes
    """
    slices = [pdcm.read_file(str(dcm)) for dcm in input_filepaths]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    if not hasattr(slices[0], "PatientWeight") or slices[0].PatientWeight is None:
        if hasattr(slices[0], "PatientsWeight"):
            patient_weight = float(slices[0].PatientsWeight)
        elif default_patient_weight is not None:
            patient_weight = default_patient_weight
        else:
            raise ValueError("Cannot compute SUV the weight is missing")
    else:
        patient_weight = float(slices[0].PatientWeight)

    # Check if all the slices come from the same serie
    same_series_uid = True
    series_uid = slices[0].SeriesInstanceUID
    for s in slices:
        same_series_uid *= series_uid == s.SeriesInstanceUID

    if not same_series_uid:
        raise RuntimeError("A slice comes from another serie")

    axial_positions = np.asarray([k.ImagePositionPatient[2] for k in slices])
    # Compute redundant slice positions
    ind2rm = [
        ind
        for ind in range(len(axial_positions))
        if axial_positions[ind] == axial_positions[ind - 1]
    ]
    # Check if there is redundancy in slice positions and remove them
    if len(ind2rm) > 0:
        slices = [k for i, k in enumerate(slices) if i not in ind2rm]
        axial_positions = np.asarray([k.ImagePositionPatient[2] for k in slices])

    slice_spacing = slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]

    pixel_spacing = np.asarray(
        [
            slices[0].PixelSpacing[0],
            slices[0].PixelSpacing[1],
            slice_spacing,
        ]
    )

    np_image = get_physical_values_pt(slices, patient_weight)

    position_final_slice = (len(slices) - 1) * slice_spacing + slices[0].ImagePositionPatient[2]
    # Test whether some slices are missing
    # due to an error at line 144: TypeError: only size-1 arrays can be converted
    # to Python scalars
    if not is_approx_equal(position_final_slice, float(slices[-1].ImagePositionPatient[2])):
        if (position_final_slice - axial_positions[-1]) / slice_spacing < 1.5:
            # If only one slice is missing
            diff = np.asarray(
                [
                    not is_approx_equal(
                        float(axial_positions[ind])
                        - float(axial_positions[ind - 1])
                        - slice_spacing,
                        0,
                    )
                    for ind in range(1, len(axial_positions))
                ]
            )
            ind2interp = int(np.where(diff)[0])
            new_slice = (np_image[:, :, ind2interp] + np_image[:, :, ind2interp + 1]) * 0.5
            new_slice = new_slice[..., np.newaxis]
            np_image = np.concatenate(
                (np_image[..., :ind2interp], new_slice, np_image[..., ind2interp:]),
                axis=2,
            )
            logger.warning("One slice is missing, we replaced it by linear interpolation")
        else:
            # if more than one slice are missing
            raise RuntimeError("Multiple slices are missing")

    image_position_patient = [float(k) for k in slices[0].ImagePositionPatient]
    sitk_image = get_sitk_volume_from_np(np_image, pixel_spacing, image_position_patient)

    sitk.WriteImage(sitk_image, str(output_filepath))


def get_sitk_volume_from_np(np_image, pixel_spacing, image_position_patient):
    """Function to get sitk volume from np image

    Args:
        np_image: extracted pet data with numpy format
        pixel_spacing: extracted pixel spacing information
        image_position_patient: extracted image position about this patient

    Returns:
        a pet sitk data
    """
    trans = (2, 0, 1)
    sitk_image = sitk.GetImageFromArray(np.transpose(np_image, trans))
    sitk_image.SetSpacing(pixel_spacing)
    sitk_image.SetOrigin(image_position_patient)
    return sitk_image


def is_approx_equal(x, y, tolerance=0.05):
    """Function to know is_approx_equal

    Args:
        x and y: two values to be compared

    Returns:
        True or False
    """
    return abs(x - y) <= tolerance


def get_physical_values_pt(slices, patient_weight):
    """Function to Get physical values from raw PET

    Args:
        slices: all pet slices of this patient
        patient_weight: a value about this patient

    Returns:
        extract physical values for pet
    """
    s = slices[0]
    if "Units" not in s:
        raise ValueError("DICOM Units tag is not found")
    units = s.Units

    if units == "BQML":

        # TODO: use the function in the dataset module to convert datetime
        acquisition_datetime = datetime.strptime(
            s.AcquisitionDate + s.AcquisitionTime.split(".")[0],
            "%Y%m%d%H%M%S",
        )
        # TODO: use the function in the dataset module to convert datetime
        series_datetime = datetime.strptime(
            s.SeriesDate + s.SeriesTime.split(".")[0],
            "%Y%m%d%H%M%S",
        )

        try:
            if datetime(1950, 1, 1) < series_datetime <= acquisition_datetime:
                scan_datetime = series_datetime
            else:
                scan_datetime_value = s.ScanDateTime
                if isinstance(scan_datetime_value, bytes):
                    scan_datetime_str = scan_datetime_value.decode("utf-8").split(".")[0]
                elif isinstance(scan_datetime_value, str):
                    scan_datetime_str = scan_datetime_value.split(".")[0]
                else:
                    raise ValueError("The value of scandatetime is not handled")
                scan_datetime = datetime.strptime(scan_datetime_str, "%Y%m%d%H%M%S")

            start_time_str = s.RadiopharmaceuticalInformationSequence[
                0
            ].RadiopharmaceuticalStartTime
            start_time = time(
                int(start_time_str[0:2]),
                int(start_time_str[2:4]),
                int(start_time_str[4:6]),
            )
            start_datetime = datetime.combine(scan_datetime.date(), start_time)
            decay_time = (scan_datetime - start_datetime).total_seconds()
        except (KeyError, AttributeError) as e:
            raise ValueError("Error calculating Decay Time") from e
        suv_results = get_suv_from_bqml(slices, decay_time, patient_weight)

    elif units == "CNTS":
        suv_results = get_suv_philips(slices)
    else:
        raise ValueError(f"The {units} units is not handled")

    return suv_results


def get_suv_philips(slices):
    """Function to Get SUV from raw PET if units = "CNTS"

    Args:
        slices: all pet slices from this patient

    Returns:
        suv philips results
    """
    image = []
    for s in slices:
        try:
            im = (float(s.RescaleSlope) * s.pixel_array + float(s.RescaleIntercept)) * float(
                s.SUVScaleFactor
            )
        except AttributeError as e:
            raise ValueError("Cannot compute SUV from raw PET for CNTS") from e

        image.append(im)

    return np.stack(image, axis=-1).astype(np.float32)


def get_suv_from_bqml(slices, decay_time, patient_weight):
    """Function to Get SUV from raw PET if units = "BQML"

    Args:
        slices: all pet slices from this patient
        decay_time: decay time
        patient_weight: patient weight information

    Returns:
        suv results
    """
    image = []
    for s in slices:
        pet = float(s.RescaleSlope) * s.pixel_array + float(s.RescaleIntercept)
        try:
            half_life = float(s.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
            total_dose = float(s.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
        except AttributeError as e:
            raise ValueError("Cannot compute SUV from raw PET for BQML") from e

        decay = 2 ** (-decay_time / half_life)
        actual_activity = total_dose * decay

        im = pet * patient_weight * 1000 / actual_activity
        image.append(im)
    return np.stack(image, axis=-1).astype(np.float32)
