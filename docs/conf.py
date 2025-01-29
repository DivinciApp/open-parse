import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'OpenParse'
copyright = '2024'
author = 'Michael Mooring'
release = '0.7.2'

extensions = [
    'sphinx.ext.autodoc',    # API documentation 
    'sphinx.ext.napoleon',   # Support for Google/NumPy docstrings
    'sphinx.ext.viewcode',   # Add links to source code
    'myst_parser',          # Support markdown files
]

# Source file parsers
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Theme settings
html_theme = 'sphinx_rtd_theme'

# Source settings
source_dir = 'docs'
root_doc = 'index'

# Output options
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# API Documentation settings
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autoclass_content = 'both'

# Include markdown files
myst_enable_extensions = [
    "colon_fence",
    "deflist"
]