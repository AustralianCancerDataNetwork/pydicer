import logging
from datetime import time, datetime
import numpy as np
import SimpleITK as sitk
import pydicom as pdcm

logger = logging.getLogger(__name__)


def convert_dicom_to_nifti_pt(
    input_filepaths,
    output_filepath,
):
    """Function to convert the dicom files contained in input_filepaths to one NIFTI image.

    Args:
        input_filepaths (list): list of the dicom paths
        output_filepath (str): path to the output file path where to store the NIFTI file.

    Raises:
        ValueError: Error to alert when the weight is missing from the PT, to compute
            the SUV.
        RuntimeError: Error to alert when one or more slices are missing
        ValueError: Raised when a modality or a unit (for the PT) is not handled.

    Returns:
        numpy.array: The numpy image, used to compute the bounding boxes
    """
    slices = [pdcm.read_file(str(dcm)) for dcm in input_filepaths]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

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

    np_image = get_physical_values_pt(slices)

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


def get_physical_values_pt(slices):
    """Function to Get physical values from raw PET

    Args:
        slices: all pet slices of this patient

    Returns:
        extract physical values for pet
    """
    s = slices[0]
    if "Units" not in s:
        raise ValueError("DICOM Units tag is not found")
    units = s.Units

    if units == "BQML":
        # Make sure we have the patient's weight
        if not hasattr(slices[0], "PatientWeight") or slices[0].PatientWeight is None:
            if hasattr(slices[0], "PatientsWeight"):
                patient_weight = float(slices[0].PatientsWeight)
            else:
                raise ValueError("Cannot compute SUV the weight is missing")
        else:
            patient_weight = float(slices[0].PatientWeight)

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
        suv_scale_factor = None
        try:
            suv_scale_factor = s.SUVScaleFactor
        except AttributeError:
            try:
                suv_scale_factor = float(s[0x7053, 0x1000].value)
            except AttributeError as exp:
                raise ValueError("Cannot compute SUV from raw PET for CNTS") from exp

        assert suv_scale_factor is not None

        im = (float(s.RescaleSlope) * s.pixel_array + float(s.RescaleIntercept)) * float(
            suv_scale_factor
        )

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

        half_life = None
        total_dose = None
        try:
            half_life = float(s.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
            total_dose = float(s.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)
        except IndexError:
            logger.warning("No RadiopharmaceuticalInformationSequence available")
        except AttributeError:
            logger.warning("Unable to read .RadionuclideHalfLife/.RadionuclideTotalDose from file")

        # Couldn't find half_life or total_dose. Try to find it in another files in the series
        if half_life is None or total_dose is None:
            for ds in slices:
                try:
                    if half_life is None:
                        half_life = float(
                            ds.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
                        )
                    if total_dose is None:
                        total_dose = float(
                            ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
                        )
                except IndexError:
                    pass
                except AttributeError:
                    pass

                if half_life is not None and total_dose is not None:
                    break

        if half_life is None or total_dose is None:
            raise ValueError("Cannot compute SUV from raw PET for BQML")

        decay = 2 ** (-decay_time / half_life)
        actual_activity = total_dose * decay

        im = pet * patient_weight * 1000 / actual_activity
        image.append(im)

    return np.stack(image, axis=-1).astype(np.float32)
