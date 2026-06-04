import sys
import os

sys.path.insert(0, os.path.abspath('../src'))

project = 'DuckSeg'
copyright = '2026, RonaLab RCNS'
author = 'DuckSeg Contributors'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
	'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build']

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

napoleon_google_docstrings = False
napoleon_numpy_docstrings = True

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
}

