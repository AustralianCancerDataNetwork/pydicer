from pydicer.input.web import WebInput


class TestInput(WebInput):
    def __init__(self, working_directory=None):
        """
        A test input class to download example data from zenodo

        Args:
            working_directory (str|pathlib.Path, optional): The working directory in which to
            store the data fetched. Defaults to a temp directory.
        """
        super().__init__(working_directory)
        self.data_url = "https://zenodo.org/record/5276878/files/HNSCC.zip"