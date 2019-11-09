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


sys.path.insert(0, os.path.abspath('..'))
import hypertunity

# The short X.Y version.
version = '.'.join(hypertunity.__version__.split('.', 2)[:2])
# The full version, including alpha/beta/rc tags.
release = hypertunity.__version__


# -- Project information -----------------------------------------------------

project = 'Hypertunity'
copyright = '2019, Georgi Dikov'
author = 'Georgi Dikov'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_keyword = True
napoleon_use_rtype = True

autodoc_typehints = 'none'


source_suffix = '.rst'


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'test*']


# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

pygments_style = 'sphinx'
add_module_names = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# this is needed as HTML5 causes an ugly rendering of the "Parameters", "Returns", etc. fields
html4_writer = True

html_theme_options = {
    "logo_only": True,
    'display_version': True,
    'style_nav_header_background': '#002A3F',
    # Toc options
    'collapse_navigation': True
}

html_context = {
    "display_github": True,     # Add 'Edit on Github' link instead of 'View page source'
    # "last_updated": True,
    # "commit": False,
}

html_logo = "_static/images/logo_inverted.svg"
html_favicon = '_static/images/favicon.ico'

github_url = "https://github.com/gdikov/hypertunity"
