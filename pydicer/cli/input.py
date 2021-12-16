from pydicer.input.test import TestInput


def testinput_cli():
    """Trigger the test input as a mini pipilien for the CLI tool
    """
    print("Running Test Input Module")
    test_input = TestInput()
    test_input.fetch_data()
