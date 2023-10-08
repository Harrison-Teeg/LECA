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
import os
import sys
autodoc_mock_imports = ["hdbscan", "GPy", "GPyOpt", 'mapie', 'ipywidgets']
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath(os.path.join('..', 'LECA')))
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------

project = 'Liquid Electrolyte Composition Analysis'
copyright = '2022, Harrison Martin'
author = 'Harrison Martin'

# The full version, including alpha/beta/rc tags
release = '0.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
        "sphinx.ext.autodoc",
        "nbsphinx",
        "sphinx.ext.autosummary",
        'sphinx.ext.coverage',
        'sphinx.ext.napoleon',
        'sphinx.ext.intersphinx',
        'numpydoc'
]

# this is needed for some reason... (from MAPIE conf.py)
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_show_class_members = False

autodoc_default_flags = ["members", "inherited-members"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# generate autosummary even if no references
autosummary_generate = True

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
# source_encoding = "utf-8-sig"

# Generate the plots for the gallery
# plot_gallery = True

# The master toctree document.
master_doc = "index"


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Syntax highlighting style
pygments_style = "sphinx"
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'classic'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']
