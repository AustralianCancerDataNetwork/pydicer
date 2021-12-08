import logging
from pathlib import Path

from pydicer.dataset import functions

logger = logging.getLogger(__name__)


class PrepareDataset:
    """
    Class that facilitates the visualisation of the data once converted

    Args:
        output_directory (str|pathlib.Path, optional): Directory in which converted data is stored.
            Defaults to ".".
    """

    def __init__(self, output_directory="."):
        self.output_directory = Path(output_directory)

    def prepare(self, target_directory, preparation_function, **kwargs):
        """
        Function to visualise the data
        """

        if isinstance(preparation_function, str):

            preparation_function = getattr(functions, preparation_function)

        if not callable(preparation_function):
            raise AttributeError(
                "preparation_function must be a function or a str defined in pydicer.dataset"
            )

        logger.info(
            "Preparing dataset in: %s using function: %s", target_directory, preparation_function
        )

        preparation_function(self.output_directory, target_directory, **kwargs)
