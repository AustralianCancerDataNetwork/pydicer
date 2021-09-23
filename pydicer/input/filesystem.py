from pydicer.input.base import InputBase


class FileSystemInput(InputBase):
    def __init__(self, directory):
        """
        Class for inputing files from the file system

        Args:
            directory (str|pathlib.Path): The directory in which to find DICOM files.
        """

        super().__init__(directory)

        if not self.working_directory.exists():
            raise FileNotFoundError("The directory provided does not exist")

        if not self.working_directory.is_dir():
            raise AttributeError("Ensure that the path specified is a directory")
