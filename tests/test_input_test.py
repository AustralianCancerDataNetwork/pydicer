from pydicer.input.test import TestInput
import os

def test_valid_input_path():
    test_input = TestInput()
    # Assert path to DICOMs exists
    assert os.path.isdir(test_input.working_directory)

def test_invalid_input_path():
    test_input = TestInput("NOT_VALID_PATH")
    # Assert path to DICOMs does not exist
    assert not os.path.isdir(test_input.working_directory)