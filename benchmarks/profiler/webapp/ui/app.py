# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gradio application interface builder for the Dynamo SLA Profiler.

This module builds the complete Gradio interface by assembling
all UI components and setting up event handlers.
"""

import gradio as gr

from benchmarks.profiler.webapp.core.constants import (
    PLOT_INTERACTION_INSTRUCTIONS,
    TABLE_CSS,
)
from benchmarks.profiler.webapp.core.orchestrator import generate_plots
from benchmarks.profiler.webapp.ui.handlers import setup_event_handlers
from benchmarks.profiler.webapp.ui.results import create_results_tabs
from benchmarks.profiler.webapp.ui.settings import (
    create_hardware_settings,
    create_model_settings,
    create_sla_settings,
)
from benchmarks.profiler.webapp.ui.tables import get_empty_tables


def build_interface(custom_js: str = None) -> gr.Blocks:
    """
    Build the complete Gradio interface for the SLA Profiler.

    Args:
        custom_js: Optional custom JavaScript to inject into the interface

    Returns:
        Configured Gradio Blocks interface
    """
    with gr.Blocks(title="Dynamo SLA Profiler", js=custom_js) as demo:
        # Header
        gr.Markdown("# Dynamo SLA Profiler")
        gr.Markdown(
            "Generate performance plots using AI Configurator to estimate profiling results. "
            "Configure the parameters below and click 'Generate Plots' to see the results."
        )
        gr.HTML(TABLE_CSS)

        # Get empty table HTML
        empty_prefill_html, empty_decode_html, empty_cost_html = get_empty_tables()

        # Store all components for event handlers
        components = {}

        with gr.Row():
            # Left panel: Settings
            with gr.Column(scale=1):
                # Model and backend settings
                gr.Markdown("### Dynamo Settings")
                model_components = create_model_settings()
                components.update(model_components)

                # Hardware settings
                gr.Markdown("### Hardware Settings")
                hardware_components = create_hardware_settings()
                components.update(hardware_components)

                # SLA settings
                gr.Markdown("### SLA Settings")
                sla_components = create_sla_settings()
                components.update(sla_components)

                # Generate button and status
                components["generate_btn"] = gr.Button(
                    "Generate Performance Plots", variant="primary", size="lg"
                )
                components["status"] = gr.Textbox(
                    label="Status",
                    value="Ready to generate plots",
                    interactive=False,
                    show_label=False,
                    lines=5,
                )

            # Right panel: Results
            with gr.Column(min_width=700):
                gr.Markdown("### Performance Results")
                gr.Markdown(PLOT_INTERACTION_INSTRUCTIONS)

                results_components = create_results_tabs(
                    empty_prefill_html, empty_decode_html, empty_cost_html
                )
                components.update(results_components)

        # Store demo reference for event handlers
        components["demo"] = demo

        # Set up all event handlers
        setup_event_handlers(components, generate_plots)

    return demo
