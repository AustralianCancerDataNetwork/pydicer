from pydicer.input.base import InputBase
from pydicer.input.base import InputBase


class FilesystemInput(InputBase):

    def __init__(self, working_directory=None):
        """
        Class for inputing files from the file system

        Args:
            working_directory (str|pathlib.Path, optional): The working directory in which to
            store the data fetched. Defaults to a temp directory.
        """

        super().__init__(working_directory)
