from pathlib import Path

from pydicer.input.test import TestInput
from pydicer.preprocess.data import PreprocessData
from pydicer.convert.data import ConvertData


def run(input_object, output_directory="."):
    """Run the pipeline to convert some data

    Args:
        input_object (InputBase): An object of a Class inheriting InputBase.
        output_directory (str, optional): Path in which to store converted data. Defaults to ".".
    """

    # Fetch the data from the Input Source
    input_object.fetch_data()

    # Preprocess the data fetch to prepare it for conversion
    preprocessed_data = PreprocessData(input_object.working_directory)
    preprocessed_result = preprocessed_data.preprocess()

    # Convert the data into the output directory
    convert_data = ConvertData(preprocessed_result, output_directory=output_directory)
    convert_data.convert()

    # TODO Visualise the converted data

    # TODO Dataset selection and preparation


def run_test(directory="./testdata"):
    """Run the pipeline using the test data provided

    Args:
        directory (str, optional): Path to store test data. Defaults to "./testdata".
    """

    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)

    working_directory = directory.joinpath("working")
    working_directory.mkdir(exist_ok=True, parents=True)
    output_directory = directory.joinpath("output")
    output_directory.mkdir(exist_ok=True, parents=True)

    test_input = TestInput(working_directory)

    run(test_input, output_directory=output_directory)


if __name__ == "__main__":
    run_test()
