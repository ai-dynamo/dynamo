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
    # Frontend migration
    "frontends/kserve": "../components/frontend/frontend_guide.html",
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
    "architecture/kv_cache_routing": "../components/router/router_guide.html",
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
    "architecture/kvbm_architecture": "../design_docs/kvbm_design.html",
    "architecture/kvbm_components": "../design_docs/kvbm_design.html",
    "architecture/kvbm_intro": "../components/kvbm/README.html",
    "architecture/kvbm_motivation": "../design_docs/kvbm_design.html",
    "architecture/kvbm_reading": "../design_docs/kvbm_design.html",
    "guides/run_kvbm_in_trtllm": "../components/kvbm/kvbm_guide.html",
    "guides/run_kvbm_in_vllm": "../components/kvbm/kvbm_guide.html",
    "guides/health_check": "../observability/health-checks.html",
    "guides/logging": "../observability/logging.html",
    "guides/metrics": "../observability/metrics.html",
    "guides/disagg_perf_tuning": "../performance/tuning.html",
    "architecture/load_planner": "../components/planner/README.html",
    "architecture/planner_intro": "../components/planner/README.html",
    "architecture/sla_planner": "../components/planner/planner_guide.html",
    "kubernetes/sla_planner_quickstart": "../components/planner/planner_guide.html",
    "guides/dynamo_run": "../reference/cli.html",
    "dynamo_glossary": "../reference/glossary.html",
    "support_matrix": "../reference/support-matrix.html",
    # Multimodal documentation consolidation (all redirect to features/multimodal/)
    "backends/vllm/multimodal": "../../features/multimodal/multimodal_vllm.html",
    "backends/vllm/multimodal_vllm_guide": "../../features/multimodal/multimodal_vllm.html",
    "backends/trtllm/multimodal_support": "../../features/multimodal/multimodal_trtllm.html",
    "backends/trtllm/multimodal_trtllm_guide": "../../features/multimodal/multimodal_trtllm.html",
    "backends/trtllm/multinode/multinode-multimodal-example": "../../../features/multimodal/multimodal_trtllm.html",
    "backends/sglang/multimodal_epd": "../../features/multimodal/multimodal_sglang.html",
    "backends/sglang/multimodal_sglang_guide": "../../features/multimodal/multimodal_sglang.html",
    "multimodal/multimodal_intro": "../features/multimodal/README.html",
    # Speculative decoding consolidation
    "backends/vllm/speculative_decoding": "../../features/speculative_decoding/speculative_decoding_vllm.html",
    # Multimodal migration to features/multimodal/
    "multimodal/index": "../features/multimodal/README.html",
    "multimodal/vllm": "../features/multimodal/multimodal_vllm.html",
    "multimodal/sglang": "../features/multimodal/multimodal_sglang.html",
    "multimodal/trtllm": "../features/multimodal/multimodal_trtllm.html",
    # Component consolidation into docs/components/
    "router/README": "../components/router/README.html",
    "router/kv_cache_routing": "../components/router/router_guide.html",
    "router/kv_events": "../integrations/kv_events_custom_engines.html",
    "planner/planner_intro": "../components/planner/README.html",
    "planner/README": "../components/planner/README.html",
    "planner/planner_guide": "../components/planner/planner_guide.html",
    "planner/planner_examples": "../components/planner/planner_examples.html",
    "planner/sla_planner_quickstart": "../components/planner/planner_guide.html",
    "planner/sla_planner": "../components/planner/planner_guide.html",
    "planner/load_planner": "../components/planner/README.html",
    "kvbm/kvbm_intro": "../components/kvbm/README.html",
    "kvbm/README": "../components/kvbm/README.html",
    "kvbm/kvbm_guide": "../components/kvbm/kvbm_guide.html",
    "kvbm/kvbm_design": "../design_docs/kvbm_design.html",
    # Profiler consolidation
    "benchmarks/sla_driven_profiling": "../components/profiler/profiler_guide.html",
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
