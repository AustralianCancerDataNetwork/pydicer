import tempfile
import logging

import abc
from pathlib import Path


logger = logging.getLogger(__name__)


class InputBase(abc.ABC):
    def __init__(self, working_directory=None):
        """
        Base class for input modules.

        Args:
            working_directory (str|pathlib.Path, optional): The working directory in which to
              store the data fetched. Defaults to a temp directory.
        """

        if working_directory is None:
            working_directory = tempfile.mkdtemp()

        self.working_directory = Path(working_directory)

        logger.debug("Working directory set to: %s", self.working_directory)
