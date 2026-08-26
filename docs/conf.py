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
html_title = 'DuckSeg Documentation'

html_theme_options = {
    "github_url": "https://github.com/ronalabrcns/DuckSeg",
    "use_edit_page_button": False,
    "navigation_with_keys": True,
    "show_toc_level": 2,
    # Logo + navbar title together; setting the classic `html_logo` alone
    # makes the theme show only the image and drop the title text.
    "logo": {
        "image_light": "_static/duckseg_logo_nobg.png",
        "image_dark": "_static/duckseg_logo_nobg.png",
        "text": html_title,
    },
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True
# Render "Attributes" sections as an inline field list rather than
# separate `.. attribute::` directives, which otherwise collide with
# autodoc's own discovery of dataclass fields ("duplicate object
# description" warnings).
napoleon_use_ivar = True

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': True,
}

# CellSAM pulls in large model weights and CUDA-only dependencies that are
# unnecessary (and often unavailable) just to build API documentation.
autodoc_mock_imports = ["cellSAM"]
