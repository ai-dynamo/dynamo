# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = "NVIDIA Dynamo"
copyright = "2024-2025, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA"

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",  # Markdown support
    "sphinx_design",  # Grid and card directives
    "sphinx.ext.autodoc",  # Auto-generate docs from docstrings
    "sphinx.ext.viewcode",  # Add source code links
    "sphinx.ext.napoleon",  # Google/NumPy style docstrings
    "sphinxcontrib.mermaid",  # Uncomment after: pip install sphinxcontrib-mermaid
]

# Handle Mermaid diagrams as code blocks (not directives) to avoid warnings
myst_fence_as_directive = ["mermaid"]  # Uncomment if sphinxcontrib-mermaid is installed

# File extensions (myst_parser automatically handles .md files)
source_suffix = [".rst", ".md"]

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",  # ::: code blocks
    "deflist",  # Definition lists
    "html_image",  # HTML images
    "tasklist",  # Task lists
]

# Templates path
templates_path = ["_templates"]

# List of patterns to ignore when looking for source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "build"]

# -- Options for HTML output -------------------------------------------------
html_theme = "alabaster"
html_static_path = ["_static"]

# Theme options
html_theme_options = {
    "description": "High-performance, low-latency inference framework",
    "github_user": "ai-dynamo",
    "github_repo": "dynamo",
    "github_button": True,
    "github_banner": True,
    "show_related": False,
    "note_bg": "#FFF59C",
}

# Document settings
master_doc = "index"
html_title = f"{project} Documentation"
html_short_title = project

# Suppress warnings for external links and missing references
suppress_warnings = [
    #'ref.doc',              # External document references
    #'myst.xref_missing',    # Missing cross-references
    #'toc.not_readable',     # Unreadable toctree entries
    #'myst.directive_unknown', # Unknown directives (like mermaid without extension)
]

# Mermaid diagram support
myst_enable_extensions.append("html_admonition")

# Additional MyST configuration
myst_heading_anchors = 3  # Generate anchors for headers
myst_substitutions = {}  # Custom substitutions
