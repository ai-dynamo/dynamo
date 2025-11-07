# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
UI components for results display in the Dynamo SLA Profiler webapp.

This module provides functions to build the results tabs with plots and tables.
"""

import gradio as gr

from benchmarks.profiler.webapp.core.constants import (
    COST_TAB_DESCRIPTION,
    DECODE_TAB_DESCRIPTION,
    PREFILL_TAB_DESCRIPTION,
)


def create_results_tabs(empty_prefill_html, empty_decode_html, empty_cost_html):
    """
    Create the results tabs with plots and tables.

    Args:
        empty_prefill_html: Empty prefill table HTML
        empty_decode_html: Empty decode table HTML
        empty_cost_html: Empty cost table HTML

    Returns:
        Dictionary of Gradio components
    """
    with gr.Tab("Prefill Performance"):
        prefill_plot = gr.Plot(
            label="Prefill Performance",
            show_label=False,
            elem_id="prefill_plot",
        )
        gr.Markdown(PREFILL_TAB_DESCRIPTION)
        gr.Markdown("#### Data Points")
        prefill_table = gr.HTML(
            value=empty_prefill_html,
            elem_id="prefill_table",
        )

    with gr.Tab("Decode Performance"):
        decode_plot = gr.Plot(
            label="Decode Performance",
            show_label=False,
            elem_id="decode_plot",
        )
        gr.Markdown(DECODE_TAB_DESCRIPTION)
        gr.Markdown("#### Data Points")
        decode_table = gr.HTML(
            value=empty_decode_html,
            elem_id="decode_table",
        )

    with gr.Tab("Cost vs SLA"):
        cost_plot = gr.Plot(
            label="Cost vs SLA",
            show_label=False,
            elem_id="cost_plot",
        )
        gr.Markdown(COST_TAB_DESCRIPTION)
        gr.Markdown("#### Data Points")
        cost_table = gr.HTML(
            value=empty_cost_html,
            elem_id="cost_table",
        )

    return {
        "prefill_plot": prefill_plot,
        "decode_plot": decode_plot,
        "cost_plot": cost_plot,
        "prefill_table": prefill_table,
        "decode_table": decode_table,
        "cost_table": cost_table,
    }
