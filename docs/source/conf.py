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
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath('/home/crivella/.virtualenvs/aiida/lib/python3.7/site-packages/'))
# import aiida_z2pack

# import glob
# if 'VIRTUAL_ENV' in os.environ:
#     site_packages_glob = os.sep.join([
#         os.environ['VIRTUAL_ENV'],
#         'lib', 'python2.7', 'site-packages', 'projectname-*py2.7.egg'])
#     site_packages = glob.glob(site_packages_glob)[-1]
#     sys.path.insert(0, site_packages)


# -- Project information -----------------------------------------------------

project = 'aiida-z2pack'
copyright = '2020, Antimo Marrazzo, Davide Grassano'
author = 'Antimo Marrazzo, Davide Grassano'

# version = '.'.join(aiida_z2pack.__version__.split('.')[:2])
# # The full version, including alpha/beta/rc tags.
# release = aiida_z2pack.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel'
]

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

# on_rtd = os.environ.get('READTHEDOCS', None) == 'True'

# if not on_rtd:  # only import and set the theme if we're building docs locally
#     try:
#         import sphinx_rtd_theme
#         html_theme = 'sphinx_rtd_theme'
#         html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
#     except ImportError:
#         # No sphinx_rtd_theme installed
#         pass
#     # Load the database environment by first loading the profile and then loading the backend through the manager
#     from aiida.manage.configuration import get_config, load_profile
#     from aiida.manage.manager import get_manager
#     config = get_config()
#     load_profile(config.default_profile_name)
#     get_manager().get_backend()
# else:
#     # Back-end settings for readthedocs online documentation.
#     from aiida.manage import configuration
#     configuration.IN_RT_DOC_MODE = True
#     configuration.BACKEND = 'django'