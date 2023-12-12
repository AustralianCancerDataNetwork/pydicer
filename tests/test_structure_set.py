# pylint: disable=redefined-outer-name,missing-function-docstring

import json

import SimpleITK as sitk

from pydicer import PyDicer
from pydicer.utils import add_structure_name_mapping, read_converted_data
from pydicer.dataset.structureset import StructureSet
from pydicer.constants import CONVERTED_DIR_NAME


def test_add_project_mapping(test_data_converted):
    working_directory = test_data_converted

    mapping_id = "test_mapping"
    mapping = {
        "Brain": ["brain", "BRAIN"],
        "SpinalCord": ["Cord", "copy_of_cord"],
    }
    add_structure_name_mapping(mapping, mapping_id=mapping_id, working_directory=working_directory)

    # Confirm that the mapping file exists
    mapping_file = working_directory.joinpath(
        ".pydicer", ".structure_set_mappings", f"{mapping_id}.json"
    )
    assert mapping_file.exists()

    # Read the file and confirm it contains the same contents as the mapping we provided
    with open(mapping_file, encoding="utf-8") as json_file:
        mapping_loaded = json.load(json_file)

    assert mapping == mapping_loaded


def test_add_structure_set_mapping(test_data_converted):
    working_directory = test_data_converted

    df = read_converted_data(working_directory)

    # Pick one structure set to supply mapping for
    struct_hash = "6d2934"
    struct_row = df[df.hashed_uid == struct_hash].iloc[0]

    mapping_id = "structure_set_mapping"
    mapping = {
        "Brain": ["brain", "BRAIN"],
        "SpinalCord": ["Cord", "copy_of_cord"],
    }
    add_structure_name_mapping(mapping, mapping_id=mapping_id, structure_set_row=struct_row)

    # Confirm that the mapping file exists
    mapping_file = working_directory.joinpath(
        CONVERTED_DIR_NAME,
        struct_row.patient_id,
        "structures",
        struct_hash,
        ".structure_set_mappings",
        f"{mapping_id}.json",
    )
    assert mapping_file.exists()

    # Read the file and confirm it contains the same contents as the mapping we provided
    with open(mapping_file, encoding="utf-8") as json_file:
        mapping_loaded = json.load(json_file)

    assert mapping == mapping_loaded


def test_add_patient_mapping(test_data_converted):
    working_directory = test_data_converted

    mapping_id = "test_mapping"
    mapping = {
        "Brain": ["brain", "BRAIN"],
        "SpinalCord": ["Cord", "copy_of_cord"],
    }
    patient_id = "HNSCC-01-0199"
    add_structure_name_mapping(
        mapping,
        mapping_id=mapping_id,
        working_directory=working_directory,
        patient_id=patient_id,
    )

    # Confirm that the mapping file exists
    mapping_file = working_directory.joinpath(
        CONVERTED_DIR_NAME,
        patient_id,
        "structures",
        ".structure_set_mappings",
        f"{mapping_id}.json",
    )
    assert mapping_file.exists()

    # Read the file and confirm it contains the same contents as the mapping we provided
    with open(mapping_file, encoding="utf-8") as json_file:
        mapping_loaded = json.load(json_file)

    assert mapping == mapping_loaded


def test_structure_set_class(test_data_converted):
    working_directory = test_data_converted

    df = read_converted_data(working_directory)

    # Pick one structure set to test mapping for
    struct_hash = "06e49c"
    struct_row = df[df.hashed_uid == struct_hash].iloc[0]

    # Check that we look up the correct structure name
    ss = StructureSet(struct_row)

    # Check that all structures are loaded
    assert len(ss.structure_names) == 38

    # Load a structure, confirm the values are as expected
    spinal_cord = ss["Cord"]
    spinal_cord_arr = sitk.GetArrayFromImage(spinal_cord)
    assert spinal_cord_arr.sum() == 7880


def test_structure_set_mapping(test_data_converted):
    working_directory = test_data_converted

    df = read_converted_data(working_directory)

    # Add a mapping
    mapping_id = "ss_mapping"
    mapping = {
        "SpinalCord": ["Cord", "copy_of_cord"],
        "Parotid_L": ["Left_parotid", "Lt_Parotid"],
        "Parotid_R": ["Right_parotid", "Rt_Parotid"],
        "Brain": ["BRAIN"],
    }
    add_structure_name_mapping(mapping, mapping_id=mapping_id, working_directory=working_directory)

    # Pick one structure set to test mapping for
    struct_hash = "06e49c"
    struct_row = df[df.hashed_uid == struct_hash].iloc[0]

    # Check that we look up the correct structure name
    ss = StructureSet(struct_row, mapping_id=mapping_id)
    assert ss.get_mapped_structure_name("SpinalCord") == "Cord"
    assert ss.get_mapped_structure_name("Parotid_L") == "Lt_Parotid"
    assert ss.get_mapped_structure_name("Parotid_R") == "Rt_Parotid"

    # Check that the correct standardised name is mapped
    assert ss.get_standardised_structure_name("Cord") == "SpinalCord"
    assert ss.get_standardised_structure_name("Lt_Parotid") == "Parotid_L"
    assert ss.get_standardised_structure_name("Rt_Parotid") == "Parotid_R"

    # Check that we can read a structure by standardised name
    spinal_cord = ss["SpinalCord"]
    spinal_cord_arr = sitk.GetArrayFromImage(spinal_cord)
    assert spinal_cord_arr.sum() == 7880

    # Check that brain is detected as not mapped for this case (as the structure isn't available)
    assert len(ss.get_unmapped_structures()) == 1
    assert ss.get_unmapped_structures()[0] == "Brain"


def test_radiomics_structure_names_standardised(test_data_converted):
    working_directory = test_data_converted
    pydicer = PyDicer(working_directory)

    # Add a mapping
    mapping_id = "rad_mapping"
    mapping = {
        "SpinalCord": ["Cord", "copy_of_cord", "cord"],
        "Parotid_L": ["Left_parotid", "Lt_Parotid", "L_parotid", "LT_Parotid"],
        "Parotid_R": ["Right_parotid", "Rt_Parotid", "R_parotid", "RT_Parotid"],
        "Brain": ["BRAIN"],
    }
    add_structure_name_mapping(mapping, mapping_id=mapping_id, working_directory=working_directory)

    # Check the radiomics without mapping
    df_radiomics = pydicer.analyse.get_all_computed_radiomics_for_dataset()
    assert len(df_radiomics.Contour.unique()) == 128

    # Check the radiomics with mapping
    df_radiomics = pydicer.analyse.get_all_computed_radiomics_for_dataset(
        structure_mapping_id=mapping_id
    )
    assert len(df_radiomics) == 13
    assert len(df_radiomics.Contour.unique()) == 4


def test_dose_metrics_structure_names_standardised(test_data_converted):
    working_directory = test_data_converted
    pydicer = PyDicer(working_directory)

    # Add a mapping
    mapping_id = "dose_mapping"
    mapping = {
        "SpinalCord": ["Cord", "copy_of_cord", "cord"],
        "Parotid_L": ["Left_parotid", "Lt_Parotid", "L_parotid", "LT_Parotid"],
        "Parotid_R": ["Right_parotid", "Rt_Parotid", "R_parotid", "RT_Parotid"],
        "Brain": ["BRAIN"],
    }
    add_structure_name_mapping(mapping, mapping_id=mapping_id, working_directory=working_directory)

    # Check the dose metrics without mapping
    df_dose_metrics = pydicer.analyse.compute_dose_metrics(d_point=[95, 50], v_point=[3])
    assert len(df_dose_metrics.label.unique()) == 128

    # Check the dose metrics with mapping
    df_dose_metrics = pydicer.analyse.compute_dose_metrics(
        d_point=[95, 50], v_point=[3], structure_mapping_id=mapping_id
    )
    assert len(df_dose_metrics) == 13
    assert len(df_dose_metrics.label.unique()) == 4
