# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Event handlers for UI interactions in the Dynamo SLA Profiler webapp.

This module sets up all event handlers for buttons, dropdowns, and other interactive elements.
"""

import gradio as gr

from benchmarks.profiler.webapp.core.constants import BACKEND_VERSIONS


def setup_event_handlers(components, generate_plots_fn):
    """
    Set up event handlers for UI interactions.

    Args:
        components: Dictionary of all UI components
        generate_plots_fn: The generate_plots function to call

    Returns:
        None (modifies components in place)
    """
    # Prepare input list for generate_plots
    inputs = [
        components["aic_model_name"],
        components["backend"],
        components["config_yaml"],
        components["use_aic"],
        components["aic_backend"],
        components["aic_backend_version"],
        components["aic_system"],
        components["min_num_gpus_per_engine"],
        components["max_num_gpus_per_engine"],
        components["num_gpus_per_node"],
        components["gpu_cost_per_hour"],
        components["isl"],
        components["osl"],
        components["max_context_length"],
        components["ttft"],
        components["itl"],
    ]

    # Prepare output list for generate_plots
    outputs = [
        components["prefill_plot"],
        components["decode_plot"],
        components["cost_plot"],
        components["status"],
        components["prefill_table"],
        components["decode_table"],
        components["cost_table"],
    ]

    # Generate button click handler
    components["generate_btn"].click(
        fn=generate_plots_fn,
        inputs=inputs,
        outputs=outputs,
    )

    # Auto-generate plots on load with default values
    components["demo"].load(
        fn=generate_plots_fn,
        inputs=inputs,
        outputs=outputs,
    )

    # Toggle AI Configurator fields visibility
    components["use_aic"].change(
        fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
        inputs=[components["use_aic"]],
        outputs=[components["aic_backend"], components["aic_backend_version"]],
    )

    # Update backend version choices when backend changes
    def update_backend_versions(backend):
        versions = BACKEND_VERSIONS.get(backend, ["1.0.0"])
        return gr.update(choices=versions, value=versions[0])

    components["aic_backend"].change(
        fn=update_backend_versions,
        inputs=[components["aic_backend"]],
        outputs=[components["aic_backend_version"]],
    )
