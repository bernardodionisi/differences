# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
from __future__ import annotations

import glob
import hashlib
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------

project = "differences"
copyright = "2022, Bernardo Dionsi"
author = "Bernardo Dionsi"

# The full version, including alpha/beta/rc tags
release = "0.1.0"

# -- General configuration ---------------------------------------------

autodoc_member_order = "bysource"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # 'sphinx.ext.viewcode',
    "sphinx_copybutton",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",  # provides links for classes in type annotations
    "nbsphinx",
    "altair.sphinxext.altairplot",
]

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# ------------------------ Options for HTML output ---------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

html_theme_options = {
    "sidebar_hide_name": True,
}
html_title = "differences"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_logo = "images/logo/bw/logo_name_bw.png"
# html_logo = "images/logo/bw/logo_bw.png"

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

# todo: ADD rst_prolog to global
#   https://stackoverflow.com/questions/9698702/how-do-i-create-a-global-role-roles-in-sphinx
# rst_prolog = """
# .. role:: underline
#     :class: underline
# """

rst_prolog = """
.. role:: underline
"""

# ----------------------------------------------------------------------

# from linearmodels: copy over notebooks from examples to docs for build
files = glob.glob("../../examples/*.ipynb")
print(files)
for file_to_copy in files:
    file_name = os.path.split(file_to_copy)[-1]

    out_dir = "notebooks"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    out_file = os.path.join(out_dir, file_name)
    existing_hash = ""
    with open(file_to_copy, "rb") as example:
        example_file = example.read()
        example_hash = hashlib.sha512(example_file).hexdigest()

    if os.path.exists(out_file):
        with open(out_file, "rb") as existing:
            existing_hash = hashlib.sha512(existing.read()).hexdigest()

    if existing_hash != example_hash:
        print(f"Copying {file_to_copy} to {out_file}")
        with open(out_file, "wb") as out:
            out.write(example_file)
