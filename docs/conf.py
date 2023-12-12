# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import os
import sys
import shutil
import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "PyDicer"
year = datetime.datetime.now().year
copyright = f"{year}, Ingham Medical Physics"
author = "Ingham Medical Physics"

# The full version, including alpha/beta/rc tags
release = "0.1.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "m2r2",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "site"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"


def setup(app):
    app.add_css_file("custom.css")


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_show_sphinx = False

# Copy in the files from the other repository directories to have them
# be rendered by Sphinx
examples_path = Path("_examples")
for notebook_path in examples_path.glob("*.ipynb"):
    os.remove(notebook_path)

examples_path.mkdir(exist_ok=True)
shutil.copy("../examples/GettingStarted.ipynb", "_examples/GettingStarted.ipynb")
shutil.copy("../examples/ConvertingData.ipynb", "_examples/ConvertingData.ipynb")
shutil.copy("../examples/VisualiseData.ipynb", "_examples/VisualiseData.ipynb")
shutil.copy("../examples/DoseMetrics.ipynb", "_examples/DoseMetrics.ipynb")
shutil.copy("../examples/Radiomics.ipynb", "_examples/Radiomics.ipynb")
shutil.copy("../examples/DatasetPreparation.ipynb", "_examples/DatasetPreparation.ipynb")
shutil.copy("../examples/WorkingWithData.ipynb", "_examples/WorkingWithData.ipynb")
shutil.copy("../examples/WorkingWithStructures.ipynb", "_examples/WorkingWithStructures.ipynb")
shutil.copy("../examples/Configuration.ipynb", "_examples/Configuration.ipynb")
shutil.copy("../examples/ObjectGeneration.ipynb", "_examples/ObjectGeneration.ipynb")
shutil.copy("../examples/AutoSegmentation.ipynb", "_examples/AutoSegmentation.ipynb")
shutil.copy("../examples/nnUNet.ipynb", "_examples/nnUNet.ipynb")
