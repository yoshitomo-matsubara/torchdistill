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
import sys
import pathlib

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())


# -- Project information -----------------------------------------------------

project = 'torchdistill'
copyright = '2023, Yoshitomo Matsubara'
author = 'Yoshitomo Matsubara'

# The full version, including alpha/beta/rc tags
import torchdistill
version = 'v' + torchdistill.__version__
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme'
]
autodoc_member_order = 'bysource'
highlight_language = 'python'

html_show_sourcelink = False
html_context = {
    'display_github': True,
    'github_user': 'yoshitomo-matsubara',
    'github_repo': 'torchdistill',
    'github_version': 'main',
    'conf_py_path': '/docs/source/'
}

html_logo = '_static/images/logo-white.png'

html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'style_external_links': True
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']