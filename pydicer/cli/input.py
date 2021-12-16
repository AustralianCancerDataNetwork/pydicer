import logging
from pydicer.input.test import TestInput


def testinput_cli(logger):
    print("Running Test Input Module")
    test_input = TestInput()
    test_input.fetch_data()
