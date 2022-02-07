import logging
from pathlib import Path

from pydicer.dataset import functions

logger = logging.getLogger(__name__)


class PrepareDataset:
    def __init__(self, working_directory="."):
        self.working_directory = Path(working_directory)

    def prepare(self, dataset_name, preparation_function, **kwargs):
        """Calls upon an appropriate preparation function to generate a clean dataset ready for
        use. Additional keyword arguments are passed through to the preparation_function.

        Args:
            dataset_name (str): The name of the dataset to generate
            preparation_function (function|str): the function use for preparation

        Raises:
            AttributeError: Raised if preparation_function is not a function or a string defining
              a known preparation function.
        """

        if isinstance(preparation_function, str):

            preparation_function = getattr(functions, preparation_function)

        if not callable(preparation_function):
            raise AttributeError(
                "preparation_function must be a function or a str defined in pydicer.dataset"
            )

        logger.info("Preparing dataset %s using function: %s", dataset_name, preparation_function)

        preparation_function(self.working_directory, dataset_name, **kwargs)
