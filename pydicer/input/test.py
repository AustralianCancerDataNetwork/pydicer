from pydicer.input.web import WebInput


class TestInput(WebInput):
    __test__ = False  # pytest will try to use this as a test class without this

    def __init__(self, working_directory=None):
        """
        A test input class to download example data from zenodo

        Args:
            working_directory (str|pathlib.Path, optional): The working directory in which to
            store the data fetched. Defaults to a temp directory.
        """

        data_url = "https://zenodo.org/record/5276878/files/HNSCC.zip"

        super().__init__(data_url, working_directory)
