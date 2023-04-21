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
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_show_sphinx = False

# Define the file extensions we want to copy
extensions_to_copy = ["md"]

# Remove any leftover files from the docs directory first
files = []
for ext in extensions_to_copy:
    files += Path(".").glob(f"**/*.{ext}")
for file in files:
    os.remove(file)

# Copy in the files from the other repository directories to have them
# be rendered by Sphinx
files = []
for ext in extensions_to_copy:
    files += Path("..").glob(f"**/*.{ext}")
for file in files:

    # Only do this for files not in the docs directory
    if file.parts[1] == "docs":
        continue

    target_file = file.relative_to("..")
    target_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(file, target_file)

shutil.rmtree("_examples", ignore_errors=True)
os.mkdir("_examples")
shutil.copy("../examples/Pipeline.ipynb", "_examples/Pipeline.ipynb")
