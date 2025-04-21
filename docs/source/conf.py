import os
import sys

sys.path.insert(0, os.path.abspath("../src/neurosnap/"))

# this is to make sure docs are built using
# the expected python executable
print(sys.executable)

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Neurosnap"
copyright = "2022-2025, Neurosnap Inc."
author = "Keaun Amani"
release = "2025.04.20"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  "sphinx.ext.intersphinx",
  "sphinx.ext.autodoc",
  "sphinx.ext.napoleon",
  "sphinx.ext.viewcode",
  "sphinx_autodoc_typehints",
]

# https://gist.github.com/bskinn/0e164963428d4b51017cebdb6cda5209
# useful for main intersphinx mappings

intersphinx_mapping = {
  "python": ("https://docs.python.org/3", None),
  "numpy": ("https://numpy.org/doc/stable/", None),
  "pandas": ("https://pandas.pydata.org/docs/", None),
  "matplotlib": ("https://matplotlib.org/stable/", None),
  "rdkit": ("https://rdkit.org/docs/", None),
}

napoleon_include_init_with_doc = True
napoleon_include_special_with_doc = True

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_favicon = "_static/favicon.ico"