import logging
import json

from pathlib import Path

from pydicer.constants import PYDICER_DIR_NAME

logger = logging.getLogger(__name__)

PYDICER_CONFIG = {
    "verbosity": {
        "module": "general",
        "description": "Level of output for standard out. Value indicates the Python built-in log "
        "level. A value of 0 (not set) will display the process bar. Logs of all levels are "
        "available in the .pydicer directory.",
        "type": int,
        "default": 0,
        "choices": [logging.NOTSET, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR],
    },
    "for_fallback_linkage": {
        "module": "general",
        "description": "Determine whether to fallback on linking objects via their Frame of "
        "Reference if no more stable link exists.",
        "type": bool,
        "default": True,
        "choices": None,
    },
    "enforce_dcm_ext": {
        "module": "preprocess",
        "description": "If True only files with the .dcm or .DCM extension will be preprocessed. "
        "otherwise any file in the DICOM directory will be preprocessed.",
        "type": bool,
        "default": True,
        "choices": None,
    },
    "interp_missing_slices": {
        "module": "convert",
        "description": "When missing slices are detected these will be interpolated if True. "
        "otherwise these cases will be sent to quarantine.",
        "type": bool,
        "default": True,
        "choices": None,
    },
    "default_patient_weight": {
        "module": "convert",
        "description": "Default patient weight to use for PET conversion if it cannot be "
        "determined from the DICOM headers. If None, those cases will be sent to "
        "quarantine.",
        "type": float,
        "default": None,
        "choices": None,
    },
    "generate_nrrd": {
        "module": "convert",
        "description": "Whether or not to generate an additional NRRD file when converting "
        "RTSTRUCT. This allows loading easily into 3D slicer.",
        "type": bool,
        "default": True,
        "choices": None,
    },
    "nrrd_colormap": {
        "module": "convert",
        "description": "Matplotlib colormap to use when saving NRRD file of structures.",
        "type": str,
        "default": "rainbow",
        "choices": None,
    },
}


class PyDicerConfig:
    class __PyDicerConfig:  # pylint: disable=invalid-name
        def __init__(self, working_dir=None):

            if working_dir is None:
                raise ValueError("working_dir must be set on config init")
            self.working_dir = Path(working_dir)

            pydicer_dir = self.working_dir.joinpath(PYDICER_DIR_NAME)
            self.config_path = pydicer_dir.joinpath("config.json")

            self.pydicer_config = {}

            if self.config_path.exists():
                # Read existing config if exists
                with open(self.config_path, "r", encoding="utf-8") as cp:
                    self.pydicer_config = json.load(cp)

            # Add config items from config object.
            # Like this if new items are added in future versions of pydicer, new config items
            # will be added in
            for key, item in PYDICER_CONFIG.items():
                if not key in self.pydicer_config:
                    self.pydicer_config[key] = item["default"]

    instance = None

    def __init__(self, working_dir=None):
        """Return the singleton instance of PyDicerConfig

        Args:
            working_dir (str|pathlib.Path, optional): The working directory for project. Required
            on first initialisation. Defaults to None.
        """

        if working_dir is not None and PyDicerConfig.instance is not None:
            # If we already have a config instance, but the working directory has changed, we will
            # recreate the instance with the new working directory.
            if not working_dir == PyDicerConfig.instance.working_dir:
                PyDicerConfig.instance = PyDicerConfig.__PyDicerConfig(working_dir)
        elif PyDicerConfig.instance is None:
            PyDicerConfig.instance = PyDicerConfig.__PyDicerConfig(working_dir)

    def get_working_dir(self):
        """Get the working directory configured for the project.

        Returns:
            pathlib.Path: The working directory
        """
        return self.instance.working_dir

    def get_config(self, name):
        """Get the value of the config item with the specified name

        Args:
            name (str): Config item name

        Raises:
            AttributeError: Config value with name doesn't exist

        Returns:
            object: Value of the config with the given name
        """

        if not name in self.instance.pydicer_config:
            raise AttributeError(f"{name} does not exist in config")

        return self.instance.pydicer_config[name]

    def set_config(self, name, value):
        """Set the value for the config with the given name

        Args:
            name (str): The name of the config to set
            value (object): The value of the config

        Raises:
            AttributeError: Config value with name doesn't exist
            ValueError: Config value is of the wrong type
        """

        if not name in self.instance.pydicer_config:
            raise AttributeError(f"{name} does not exist in config")

        if not isinstance(value, PYDICER_CONFIG[name]["type"]) and not value is None:
            raise ValueError(
                f"Config {name} must be of type " f"{type(self.instance.pydicer_config[name])}"
            )

        self.instance.pydicer_config[name] = value
        self.save_config()

    def save_config(self):
        """Save the config to the pydicer directory"""

        if not self.instance.config_path.parent.exists():
            self.instance.config_path.parent.mkdir()

        with open(self.instance.config_path, "w", encoding="utf-8") as fp:
            json.dump(self.instance.pydicer_config, fp, indent=2)
