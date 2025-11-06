# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Main entry point for the Dynamo SLA Profiler web application.

This webapp provides an interactive interface for profiling LLM inference performance
using AI Configurator estimates.
"""

from benchmarks.profiler.webapp.ui.app import build_interface


def main():
    """Launch the Dynamo SLA Profiler webapp."""
    # Load custom JavaScript for enhanced interactivity
    with open("benchmarks/profiler/webapp/static/utils.js", "r") as f:
        custom_js = f"()=>{{{f.read()}}}"

    # Build and launch the interface
    demo = build_interface(custom_js)
    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    main()
