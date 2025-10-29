# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
import os
import sys

# -- Project information -----------------------------------------------------
project = "NVIDIA Dynamo"
copyright = "2024-2025, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA"
release = "latest"

# -- General configuration ---------------------------------------------------

# Standard extensions
extensions = [
    "ablog",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_prompt",
    # "sphinxcontrib.bibtex",
    "sphinx_tabs.tabs",
    "sphinx_sitemap",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.ifconfig",
    "sphinx.ext.extlinks",
    "sphinxcontrib.mermaid",
    "sphinx_reredirects",
]

# Redirects configuration
redirects = {
    "guides/tool-calling.html": "../agents/tool-calling.html",  # relative
    "architecture/architecture.html": "../design_docs/architecture.html",  # relative
    "architecture/disagg_serving.html": "../design_docs/disagg_serving.html",  # relative
    "architecture/distributed_runtime.html": "../design_docs/distributed_runtime.html",  # relative
    "architecture/dynamo_flow.html": "../design_docs/dynamo_flow.html",  # relative
    "architecture/request_cancellation.html": "../fault_tolerance/request_cancellation.html",  # relative
    "architecture/request_migration.html": "../fault_tolerance/request_migration.html",  # relative
    "kubernetes/create_deployment.html": "../kubernetes/deployment/create_deployment.html",  # relative
    "kubernetes/minikube.html": "../kubernetes/deployment/minikube.html",  # relative
    "kubernetes/multinode-deployment.html": "../kubernetes/deployment/multinode-deployment.html",  # relative
    "kubernetes/logging.html": "../kubernetes/observability/logging.html",  # relative
    "kubernetes/metrics.html": "../kubernetes/observability/metrics.html",  # relative
    "architecture/kv_cache_routing.html": "../router/kv_cache_routing.html",  # relative
}

# Custom extensions
sys.path.insert(0, os.path.abspath("_extensions"))
extensions.append("github_alerts")

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
html_theme = "nvidia_sphinx_theme"
html_static_path = ["_static"]
html_extra_path = ["project.json", "versions1.json"]
html_theme_options = {
    "collapse_navigation": False,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ai-dynamo/dynamo",
            "icon": "fa-brands fa-github",
        }
    ],
    "switcher": {
        "json_url": "versions1.json",
        "version_match": release,
    },
    "extra_head": {
        """
    <script src="https://assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js" ></script>
    """
    },
    "extra_footer": {
        """
    <script type="text/javascript">if (typeof _satellite !== "undefined") {_satellite.pageBottom();}</script>
    """
    },
    "navbar_start": ["navbar-logo"],
    "primary_sidebar_end": [],
}

# Document settings
master_doc = "index"
html_title = f"{project} Documentation"
html_short_title = project
html_baseurl = "https://docs.nvidia.com/dynamo/latest/"

# Suppress warnings for external links and missing references
suppress_warnings = [
    "myst.xref_missing",  # Missing cross-references of relative links outside docs folder
]

# Additional MyST configuration
myst_heading_anchors = 7  # Generate anchors for headers
myst_substitutions = {}  # Custom substitutions
