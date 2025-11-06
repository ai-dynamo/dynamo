# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Interactive plotting functions for Gradio webapp using Plotly.

This module provides interactive versions of the profiler plots using Plotly,
which integrates seamlessly with Gradio's gr.Plot component.
"""

import numpy as np
import plotly.graph_objects as go

from benchmarks.profiler.utils.parato import compute_parato
from benchmarks.profiler.webapp.core.constants import PLOTLY_COLORS, PLOTLY_DARK_THEME


def _configure_dark_theme(fig, title, xaxis_title, yaxis_title):
    """
    Apply dark theme configuration to a Plotly figure.

    Args:
        fig: Plotly Figure object
        title: Plot title
        xaxis_title: X-axis title
        yaxis_title: Y-axis title
    """
    fig.update_layout(
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 18 if len(title) < 60 else 16},
        },
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovermode="closest",
        showlegend=True,
        autosize=True,
        clickmode="event+select",  # Enable click selection
        **PLOTLY_DARK_THEME,
    )

    # Add grid
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.3)")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(128, 128, 128, 0.3)")


def _add_target_line(fig, target_value, label, max_y):
    """
    Add a target reference line to a plot.

    Args:
        fig: Plotly Figure object
        target_value: X-coordinate of the vertical line
        label: Label for the target line
        max_y: Maximum Y value for the line
    """
    fig.add_trace(
        go.Scatter(
            x=[target_value, target_value],
            y=[0, max_y * 1.1],
            mode="lines",
            line=dict(color="red", width=2, dash="dash"),
            name=label,
            hovertemplate=f"{label}<extra></extra>",
        )
    )


def _configure_selection_style(fig, mode, selected_color="red", selected_size=16):
    """
    Configure selection appearance for interactive plots.

    Args:
        fig: Plotly Figure object
        mode: Trace mode (e.g., "markers+text", "lines+markers")
        selected_color: Color for selected markers
        selected_size: Size for selected markers
    """
    fig.update_traces(
        selected=dict(marker=dict(color=selected_color, size=selected_size)),
        unselected=dict(marker=dict(opacity=0.4 if "text" in mode else 0.5)),
        selector=dict(mode=mode),
    )


def plot_prefill_performance_interactive(
    prefill_results: tuple, target_ttft: float
) -> go.Figure:
    """
    Create interactive Plotly plot for prefill performance.

    Args:
        prefill_results: Tuple of (num_gpus_list, ttft_list, thpt_per_gpu_list)
        target_ttft: Target TTFT in milliseconds (for reference line)

    Returns:
        Plotly Figure object for Gradio gr.Plot
    """
    num_gpus_list, ttft_list, thpt_per_gpu_list = prefill_results

    fig = go.Figure()

    # Add scatter plot for data points with custom data
    fig.add_trace(
        go.Scatter(
            x=ttft_list,
            y=thpt_per_gpu_list,
            mode="markers+text",
            marker=dict(size=12, color="blue", line=dict(width=2, color="darkblue")),
            text=[f"{n} GPU(s)" for n in num_gpus_list],
            textposition="top center",
            textfont=dict(size=10),
            name="GPU Configurations",
            hovertemplate="<b>%{text}</b><br>"
            + "TTFT: %{x:.2f} ms<br>"
            + "Throughput: %{y:.2f} tokens/s/GPU<br>"
            + "<extra></extra>",
            customdata=list(zip(num_gpus_list, ttft_list, thpt_per_gpu_list)),
        )
    )

    # Add target TTFT line
    max_thpt = max(thpt_per_gpu_list) if thpt_per_gpu_list else 1000
    _add_target_line(fig, target_ttft, f"Target TTFT: {target_ttft} ms", max_thpt)

    # Apply dark theme and configure layout
    _configure_dark_theme(
        fig,
        "Prefill Performance",
        "Time to First Token (ms)",
        "Prefill Throughput per GPU (tokens/s/GPU)",
    )

    # Configure selection appearance
    _configure_selection_style(
        fig, "markers+text", selected_color="red", selected_size=16
    )

    return fig


def plot_decode_performance_interactive(
    decode_results: list, target_itl: float
) -> go.Figure:
    """
    Create interactive Plotly plot for decode performance.

    Args:
        decode_results: List of tuples (num_gpus, itl_list, thpt_per_gpu_list)
        target_itl: Target ITL in milliseconds (for reference line)

    Returns:
        Plotly Figure object for Gradio gr.Plot
    """
    fig = go.Figure()

    # Plot each GPU configuration
    for idx, (num_gpus, itl_list, thpt_per_gpu_list) in enumerate(decode_results):
        color = PLOTLY_COLORS[idx % len(PLOTLY_COLORS)]
        # Prepare custom data for each point
        customdata = [
            [num_gpus, itl, thpt] for itl, thpt in zip(itl_list, thpt_per_gpu_list)
        ]

        fig.add_trace(
            go.Scatter(
                x=itl_list,
                y=thpt_per_gpu_list,
                mode="lines+markers",
                marker=dict(size=8, color=color),
                line=dict(color=color, width=2),
                name=f"{num_gpus} GPU(s)",
                hovertemplate=f"<b>{num_gpus} GPU(s)</b><br>"
                + "ITL: %{x:.2f} ms<br>"
                + "Throughput: %{y:.2f} tokens/s/GPU<br>"
                + "<extra></extra>",
                customdata=customdata,
            )
        )

    # Add target ITL line
    all_thpt = [
        thpt for _, _, thpt_list in decode_results for thpt in thpt_list if thpt_list
    ]
    max_thpt = max(all_thpt) if all_thpt else 1000
    _add_target_line(fig, target_itl, f"Target ITL: {target_itl} ms", max_thpt)

    # Apply dark theme and configure layout
    _configure_dark_theme(
        fig,
        "Decode Performance",
        "Inter Token Latency (ms)",
        "Decode Throughput per GPU (tokens/s/GPU)",
    )

    # Configure selection appearance for markers
    _configure_selection_style(
        fig, "lines+markers", selected_color="yellow", selected_size=12
    )

    return fig


def plot_cost_sla_interactive(
    isl: int,
    osl: int,
    prefill_results: tuple,
    decode_results: list,
    gpu_cost_per_hour: float = 3.0,
) -> go.Figure:
    """
    Create interactive Plotly plot for cost vs SLA analysis.

    Args:
        isl: Input sequence length
        osl: Output sequence length
        prefill_results: Tuple of (num_gpus, ttft, thpt_per_gpu) for prefill
        decode_results: List of tuples (num_gpus, itl_list, thpt_per_gpu_list) for decode
        gpu_cost_per_hour: Cost per GPU per hour in dollars (default: 3.0)

    Returns:
        Plotly Figure object for Gradio gr.Plot
    """
    # Compute Pareto fronts
    p_ttft, p_thpt = compute_parato(prefill_results[1], prefill_results[2])

    _d_itl, _d_thpt = [], []
    for _d_result in decode_results:
        _d_itl.extend(_d_result[1])
        _d_thpt.extend(_d_result[2])
    d_itl, d_thpt = compute_parato(_d_itl, _d_thpt)

    # Convert to numpy arrays for element-wise operations
    p_ttft = np.array(p_ttft)
    p_thpt = np.array(p_thpt)
    d_itl = np.array(d_itl)
    d_thpt = np.array(d_thpt)

    # Calculate cost metrics
    fig = go.Figure()

    for idx, (_p_ttft, _p_thpt) in enumerate(zip(p_ttft, p_thpt)):
        # Calculate costs for this TTFT curve
        prefill_cost = isl * 1000 / _p_thpt * gpu_cost_per_hour / 3600

        # Calculate tokens per user and cost arrays (element-wise operations)
        tokens_per_user_array = 1000 / d_itl  # Element-wise division with numpy array
        cost_array = osl * 1000 / d_thpt * gpu_cost_per_hour / 3600 + prefill_cost

        color = PLOTLY_COLORS[idx % len(PLOTLY_COLORS)]

        # Prepare custom data for each point
        customdata = [
            [
                _p_ttft,
                _p_thpt,
                float(d_itl[i]),
                float(d_thpt[i]),
                float(tokens_per_user_array[i]),
                float(cost_array[i]),
            ]
            for i in range(len(d_itl))
        ]

        # Add line plot for this TTFT curve
        fig.add_trace(
            go.Scatter(
                x=tokens_per_user_array,
                y=cost_array,
                mode="lines+markers",
                marker=dict(size=10, symbol="x", color=color, line=dict(width=2)),
                line=dict(color=color, width=2),
                name=f"TTFT: {_p_ttft:.2f}ms",
                hovertemplate=f"<b>TTFT: {_p_ttft:.2f}ms</b><br>"
                + "Tokens/User: %{x:.2f}<br>"
                + "Cost: $%{y:.3f}<br>"
                + "<extra></extra>",
                customdata=customdata,
            )
        )

    # Apply dark theme and configure layout
    title = f"Cost Per 1000 i{isl}o{osl} requests (GPU/hour = ${gpu_cost_per_hour:.2f}) Under Different SLA"
    _configure_dark_theme(fig, title, "Tokens per User", "Cost ($)")

    # Configure selection appearance for markers
    _configure_selection_style(
        fig, "lines+markers", selected_color="yellow", selected_size=14
    )

    return fig
