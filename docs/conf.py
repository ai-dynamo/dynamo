# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Configuration file for the Sphinx documentation builder.
import os
import sys

# -- Project information -----------------------------------------------------
project = "NVIDIA Dynamo"
copyright = "2024-2026, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA"

# Version is set via DYNAMO_DOCS_VERSION env var during build (e.g., "0.3.0")
# Defaults to "dev" for main branch and PR builds
release = os.environ.get("DYNAMO_DOCS_VERSION", "dev")

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
    # PR  #3802
    "guides/tool-calling": "../agents/tool-calling.html",  # key format corrected
    "architecture/architecture": "../design_docs/architecture.html",
    "architecture/disagg_serving": "../design_docs/disagg_serving.html",
    "architecture/distributed_runtime": "../design_docs/distributed_runtime.html",
    "architecture/dynamo_flow": "../design_docs/dynamo_flow.html",
    "architecture/request_cancellation": "../fault_tolerance/request_cancellation.html",
    "architecture/request_migration": "../fault_tolerance/request_migration.html",
    "kubernetes/create_deployment": "../kubernetes/deployment/create_deployment.html",
    "kubernetes/minikube": "../kubernetes/deployment/minikube.html",
    "kubernetes/multinode-deployment": "../kubernetes/deployment/multinode-deployment.html",
    "kubernetes/logging": "../kubernetes/observability/logging.html",
    "kubernetes/metrics": "../kubernetes/observability/metrics.html",
    "architecture/kv_cache_routing": "../router/kv_cache_routing.html",
    # PR #3658
    "API/nixl_connect/README": "../../api/nixl_connect/README.html",
    "API/nixl_connect/connector": "../../api/nixl_connect/connector.html",
    "API/nixl_connect/descriptor": "../../api/nixl_connect/descriptor.html",
    "API/nixl_connect/device": "../../api/nixl_connect/device.html",
    "API/nixl_connect/device_kind": "../../api/nixl_connect/device_kind.html",
    "API/nixl_connect/operation_status": "../../api/nixl_connect/operation_status.html",
    "API/nixl_connect/rdma_metadata": "../../api/nixl_connect/rdma_metadata.html",
    "API/nixl_connect/read_operation": "../../api/nixl_connect/read_operation.html",
    "API/nixl_connect/readable_operation": "../../api/nixl_connect/readable_operation.html",
    "API/nixl_connect/writable_operation": "../../api/nixl_connect/writable_operation.html",
    "API/nixl_connect/write_operation": "../../api/nixl_connect/write_operation.html",
    "guides/backend": "../development/backend-guide.html",
    "runtime/README": "../development/runtime-guide.html",
    "guides/tool_calling": "../agents/tool-calling.html",
    "architecture/kvbm_architecture": "../kvbm/kvbm_architecture.html",
    "architecture/kvbm_components": "../kvbm/kvbm_components.html",
    "architecture/kvbm_intro": "../kvbm/kvbm_intro.html",
    "architecture/kvbm_motivation": "../kvbm/kvbm_motivation.html",
    "architecture/kvbm_reading": "../kvbm/kvbm_reading.html",
    "guides/run_kvbm_in_trtllm": "../kvbm/trtllm-setup.html",
    "guides/run_kvbm_in_vllm": "../kvbm/vllm-setup.html",
    "guides/health_check": "../observability/health-checks.html",
    "guides/logging": "../observability/logging.html",
    "guides/metrics": "../observability/metrics.html",
    "guides/disagg_perf_tuning": "../performance/tuning.html",
    "architecture/load_planner": "../planner/load_planner.html",
    "architecture/planner_intro": "../planner/planner_intro.html",
    "architecture/sla_planner": "../planner/sla_planner.html",
    "kubernetes/sla_planner_quickstart": "../planner/sla_planner_quickstart.html",
    "guides/dynamo_run": "../reference/cli.html",
    "dynamo_glossary": "../reference/glossary.html",
    "support_matrix": "../reference/support-matrix.html",
    "components/router/README": "../router/README.html",
    # Multimodal documentation consolidation
    "backends/vllm/multimodal": "../../multimodal/vllm.html",
    "backends/vllm/multimodal_vllm_guide": "../../multimodal/vllm.html",
    "backends/trtllm/multimodal_support": "../../multimodal/trtllm.html",
    "backends/trtllm/multimodal_trtllm_guide": "../../multimodal/trtllm.html",
    "backends/trtllm/multinode/multinode-multimodal-example": "../../../multimodal/trtllm.html",
    "backends/sglang/multimodal_epd": "../../multimodal/sglang.html",
    "backends/sglang/multimodal_sglang_guide": "../../multimodal/sglang.html",
    "multimodal/multimodal_intro": "index.html",
    # Speculative decoding consolidation
    "backends/vllm/speculative_decoding": "../../features/speculative_decoding/speculative_decoding_vllm.html",
    # Multimodal migration to features/multimodal/
    "multimodal/index": "../features/multimodal/README.html",
    "multimodal/vllm": "../features/multimodal/multimodal_vllm.html",
    "multimodal/sglang": "../features/multimodal/multimodal_sglang.html",
    "multimodal/trtllm": "../features/multimodal/multimodal_trtllm.html",
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
html_extra_path = ["project.json"]
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
        # Use single shared URL so all versions see the same switcher list
        # When a new version is added, all old docs automatically see it
        "json_url": "https://docs.nvidia.com/dynamo/versions1.json",
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
