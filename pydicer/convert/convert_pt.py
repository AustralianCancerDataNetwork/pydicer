import warnings
from datetime import time, datetime
import numpy as np
import SimpleITK as sitk
import pydicom as pdcm
from pydicom.tag import Tag


def convert_dicom_to_nifty_pt(
        input_filepaths,
        output_filepath,
        modality="PT",
        sitk_writer=None,
        patient_weight_from_ct=None,
        dtype_image=np.float32,
    ):
    """Function to convert the dicom files contained in input_filepaths to one
       NIFTI image.
    Args:
        input_filepaths (list): list of the dicom paths
        output_filepath (str): path to the output file path where to store the
                             NIFTI file.
        modality (str, optional): The modality of the DICOM, it is used to
                                  obtain the correct physical values
                                  (Hounsfield unit for the CT and SUV for the
                                  PT). Defaults to 'CT'.
        sitk_writer (sitk.WriteImage(), optional): The SimpleITK object used
                                                   to write an array to the
                                                   NIFTI format. Defaults to
                                                   None.
        patient_weight_from_ct (float, optional): If the patient's weight is
                                                  missing from the PT DICOM
                                                  it can be provided through
                                                  this argument. Defaults to
                                                  None.
        dtype_image (numpy.dtype, optional): The dtype in which to save the
                                             image. Defaults to np.float32.
    Raises:
        MissingWeightException: Error to alert when the weight is missing from
                                the PT, to compute the SUV.
        RuntimeError: Error to alert when one or more slices are missing
        ValueError: Raised when a modality or a unit (for the PT) is not
                    handled.
    Returns:
        numpy.array: The numpy image, used to compute the bounding boxes
    """
    slices = [pdcm.read_file(str(dcm["path"])) for dcm in input_filepaths]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    if modality == "PT":
        if hasattr(slices[0], "PatientWeight") and (slices[0].PatientWeight is None):
            if hasattr(slices[0], "PatientsWeight"):
                patient_weight = float(slices[0].PatientsWeight)
            elif patient_weight_from_ct is not None:
                patient_weight = patient_weight_from_ct
            else:
                # raise MissingWeightException(
                #     'Cannot compute SUV the weight is missing')  # from Shuchao's code
                patient_weight = 75.0  # From Shuchao's code (default)
                warnings.warn(
                    "Cannot find the weight of the patient, hence it "
                    "is approximated to be 75.0 kg"
                )
        elif not hasattr(slices[0], "PatientWeight"):
            if hasattr(slices[0], "PatientsWeight"):
                patient_weight = float(slices[0].PatientsWeight)
            elif patient_weight_from_ct is not None:
                patient_weight = patient_weight_from_ct
            else:
                # raise MissingWeightException(
                #     'Cannot compute SUV the weight is missing')  # from Shuchao's code
                patient_weight = 75.0  # From Shuchao's code (default)
                warnings.warn(
                    "Cannot find the weight of the patient, hence it "
                    "is approximated to be 75.0 kg"
                )
        else:
            patient_weight = float(slices[0].PatientWeight)

    # Check if all the slices come from the same serie
    same_serie_uid = True
    serie_uid = slices[0].SeriesInstanceUID
    for s in slices:
        same_serie_uid *= serie_uid == s.SeriesInstanceUID

    if not same_serie_uid:
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

    slice_spacing = (
        slices[1].ImagePositionPatient[2] - slices[0].ImagePositionPatient[2]
    )

    pixel_spacing = np.asarray(
        [
            slices[0].PixelSpacing[0],
            slices[0].PixelSpacing[1],
            slice_spacing,
        ]
    )

    if modality == "PT":
        np_image = get_physical_values_pt(slices, patient_weight, dtype=dtype_image)
    else:
        raise ValueError("The modality {} is not supported".format(modality))

    position_final_slice = (len(slices) - 1) * slice_spacing + slices[
        0
    ].ImagePositionPatient[2]
    # Test whether some slices are missing
    # Pang -- due to an error at line 144: TypeError: only size-1 arrays can be converted
    # to Python scalars
    if not is_approx_equal(
            position_final_slice, float(slices[-1].ImagePositionPatient[2])
        ):
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
            new_slice = (
                np_image[:, :, ind2interp] + np_image[:, :, ind2interp + 1]
            ) * 0.5
            new_slice = new_slice[..., np.newaxis]
            np_image = np.concatenate(
                (np_image[..., :ind2interp], new_slice, np_image[..., ind2interp:]),
                axis=2,
            )
            warnings.warn(
                "One slice is missing, we replaced it by linear interpolation"
            )
        else:
            # if more than one slice are missing
            raise RuntimeError("Multiple slices are missing")

    image_position_patient = [float(k) for k in slices[0].ImagePositionPatient]
    sitk_image = get_sitk_volume_from_np(
        np_image, pixel_spacing, image_position_patient
    )

    sitk_writer.SetFileName(output_filepath)
    sitk_writer.Execute(sitk_image)

    return np.transpose(np_image, (1, 0, 2)), pixel_spacing, image_position_patient


def get_sitk_volume_from_np(np_image, pixel_spacing, image_position_patient):
    """Function to get sitk volume from np image
    Args:
    Returns:
    """
    trans = (2, 0, 1)
    sitk_image = sitk.GetImageFromArray(np.transpose(np_image, trans))
    sitk_image.SetSpacing(pixel_spacing)
    sitk_image.SetOrigin(image_position_patient)
    return sitk_image


def is_approx_equal(x, y, tolerance=0.05):
    """Function to know is_approx_equal
    Args:
    Returns:
    """
    return abs(x - y) <= tolerance


class MissingWeightException(RuntimeError):
    pass


def get_physical_values_pt(slices, patient_weight, dtype=np.float32):
    """Function to Get physical values from raw PET
    Args:
    Returns:
    """
    s = slices[0]
    # units = s.Units
    if "Units" in s:  # from Shuchao's code (default)
        units = s.Units
    else:
        units = "CNTS"

    if units == "BQML":

        acquisition_datetime = datetime.strptime(
            s[Tag(0x00080022)].value + s[Tag(0x00080032)].value.split(".")[0],
            "%Y%m%d%H%M%S",
        )
        serie_datetime = datetime.strptime(
            s[Tag(0x00080021)].value + s[Tag(0x00080031)].value.split(".")[0],
            "%Y%m%d%H%M%S",
        )

        try:
            if datetime(1950, 1, 1) < serie_datetime <= acquisition_datetime:
                scan_datetime = serie_datetime
            else:
                scan_datetime_value = s[Tag(0x0009100D)].value
                if isinstance(scan_datetime_value, bytes):
                    scan_datetime_str = scan_datetime_value.decode("utf-8").split(".")[
                        0
                    ]
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
        except KeyError:
            warnings.warn(
                "Estimation of time decay for SUV"
                " computation from average parameters"
            )
            decay_time = 1.75 * 3600  # From Martin's code
        except AttributeError:
            warnings.warn(
                "'Dataset' object has no attribute 'RadiopharmaceuticalStartTime'"
            )
            decay_time = 1.75 * 3600  # From Shuchao's code
        suv_results = get_suv_from_bqml(slices, decay_time, patient_weight, dtype=dtype)

    elif units == "CNTS":
        suv_results = get_suv_philips(slices, dtype=dtype)
    else:
        raise ValueError("The {} units is not handled".format(units))

    return suv_results


def get_suv_philips(slices, dtype=np.float32):
    """Function to Get SUV from raw PET if units = "CNTS"
    Args:
    Returns:
    """
    image = list()
    suv_scale_factor_tag = Tag(0x70531000)
    for s in slices:
        if (
                suv_scale_factor_tag in s
                and "RescaleSlope" in s
                and "RescaleIntercept" in s
            ):  # # From Shuchao's code
            im = (
                float(s.RescaleSlope) * s.pixel_array + float(s.RescaleIntercept)
            ) * float(s[suv_scale_factor_tag].value)
        else:
            im = (1.0 * s.pixel_array + 0.0) * 0.000587

        image.append(im)

    return np.stack(image, axis=-1).astype(dtype)


def get_suv_from_bqml(slices, decay_time, patient_weight, dtype=np.float32):
    """Function to Get SUV from raw PET if units = "BQML"
    Args:
    Returns:
    """
    image = list()
    for s in slices:
        pet = float(s.RescaleSlope) * s.pixel_array + float(s.RescaleIntercept)
        if "RadionuclideHalfLife" in s.RadiopharmaceuticalInformationSequence[0]:
            half_life = float(
                s.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife
            )
        else:
            half_life = 6500.0  # # From Shuchao's code (default)
            print("there is no RadionuclideHalfLife")
        if "RadionuclideTotalDose" in s.RadiopharmaceuticalInformationSequence[0]:
            total_dose = float(
                s.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose
            )
        else:
            total_dose = 487254240.0  # # From Shuchao's code (default)
            print("there is no RadionuclideTotalDose")

        decay = 2 ** (-decay_time / half_life)
        actual_activity = total_dose * decay

        im = pet * patient_weight * 1000 / actual_activity
        image.append(im)
    return np.stack(image, axis=-1).astype(dtype)
