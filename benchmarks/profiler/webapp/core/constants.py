# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Constants and configuration for the Dynamo SLA Profiler webapp.
"""

# Table headers for different performance metrics
PREFILL_TABLE_HEADERS = [
    "GPUs",
    "TTFT (ms)",
    "Throughput (tokens/s/GPU)",
]

DECODE_TABLE_HEADERS = [
    "GPUs",
    "ITL (ms)",
    "Throughput (tokens/s/GPU)",
]

COST_TABLE_HEADERS = [
    "TTFT (ms)",
    "Prefill Thpt (tokens/s/GPU)",
    "ITL (ms)",
    "Decode Thpt (tokens/s/GPU)",
    "Tokens/User",
    "Cost ($)",
]

# Backend version mapping
BACKEND_VERSIONS = {
    "trtllm": ["1.0.0", "0.20.0", "0.19.0", "0.18.0"],
    "vllm": ["0.10.0"],
    "sglang": ["0.4.5"],
}

# Supported GPU systems
GPU_SYSTEMS = [
    "H100_SXM",
    "H200_SXM",
    "A100_SXM",
    "A100_PCIE",
]

# Supported inference backends
INFERENCE_BACKENDS = ["vllm", "sglang", "trtllm"]

# GPU count options
MIN_GPU_OPTIONS = [1, 2, 4, 8]
MAX_GPU_OPTIONS = [1, 2, 4, 8, 16]

# Default decode interpolation granularity
DEFAULT_DECODE_INTERPOLATION_GRANULARITY = 6

# CSS styles for custom table rendering
TABLE_CSS = """
<style>
    .dynamo-table-wrapper {
        overflow-x: auto;
        margin-top: 0.5rem;
    }
    .dynamo-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
    }
    .dynamo-table thead {
        background: rgba(255, 255, 255, 0.05);
        text-transform: uppercase;
        letter-spacing: 0.02em;
    }
    .dynamo-table th,
    .dynamo-table td {
        padding: 0.55rem 0.75rem;
        text-align: left;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .dynamo-table tbody tr:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    .dynamo-table-empty {
        text-align: center;
        padding: 0.85rem 0;
        opacity: 0.7;
    }
</style>
"""

# Default configuration YAML placeholder
DEFAULT_CONFIG_YAML = """apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-disagg
spec:
  services:
    Frontend:
      dynamoNamespace: vllm-disagg
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag"""

# Plot interaction instructions
PLOT_INTERACTION_INSTRUCTIONS = """
**How to interact with plots:**
- **Hover** over points to see detailed information
- **Click** points to select them (click again to deselect)
- **Multiple selection**: Click multiple points with shift key or select tools from the top right corner to compare specific configurations
- The table below each plot will filter to show only selected points, or all points if none are selected
"""

# Tab descriptions
PREFILL_TAB_DESCRIPTION = """
**Prefill Performance**: Interactive plot showing the relationship between Time to First Token (TTFT)
and throughput per GPU for different GPU counts. **Click points to select/deselect** (multi-select enabled).
Table shows selected points, or all points if none selected.
"""

DECODE_TAB_DESCRIPTION = """
**Decode Performance**: Interactive plot showing the relationship between Inter Token Latency (ITL)
and throughput per GPU for different GPU counts. **Click points to select/deselect** (multi-select enabled).
Table shows selected points, or all points if none selected.
"""

COST_TAB_DESCRIPTION = """
**Cost Analysis**: Interactive plot showing the cost per 1000 requests under different SLA configurations.
Lower curves represent better cost efficiency for the same throughput. **Click points to select/deselect** (multi-select enabled).
Table shows selected points, or all points if none selected.
"""

# Plotly color palette
PLOTLY_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

# Plotly dark theme configuration
PLOTLY_DARK_THEME = {
    "template": "plotly_dark",
    "plot_bgcolor": "rgba(0, 0, 0, 0)",
    "paper_bgcolor": "rgba(0, 0, 0, 0)",
    "modebar": dict(
        bgcolor="rgba(0, 0, 0, 0)",
        color="rgba(255, 255, 255, 0.5)",
        activecolor="rgba(255, 255, 255, 0.9)",
    ),
    "legend": dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(0, 0, 0, 0.5)",
        font=dict(color="white"),
    ),
}
