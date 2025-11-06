# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Table building and data preparation utilities for the Dynamo SLA Profiler webapp.
"""

from numbers import Real

import numpy as np

from benchmarks.profiler.utils.parato import compute_parato
from benchmarks.profiler.webapp.core.constants import (
    COST_TABLE_HEADERS,
    DECODE_TABLE_HEADERS,
    PREFILL_TABLE_HEADERS,
)


def _format_cell(value):
    """Format a cell value for display in HTML table."""
    if isinstance(value, bool):
        return "✅" if value else "❌"
    if isinstance(value, Real) and not isinstance(value, bool):
        if isinstance(value, int):
            return f"{value}"
        return f"{value:.3f}"
    return str(value)


def build_table_html(headers, rows):
    """
    Build an HTML table from headers and rows.

    Args:
        headers: List of header strings
        rows: List of row data (each row is a list of values)

    Returns:
        HTML string containing the table
    """
    header_html = "".join(f"<th>{header}</th>" for header in headers)

    if not rows:
        empty_row = (
            f"<tr><td class='dynamo-table-empty' colspan='{len(headers)}'>"
            "No data selected yet. Click points on the plot to populate this table."
            "</td></tr>"
        )
        body_html = empty_row
    else:
        body_html = "".join(
            "<tr>" + "".join(f"<td>{_format_cell(cell)}</td>" for cell in row) + "</tr>"
            for row in rows
        )

    return (
        "<div class='dynamo-table-wrapper'>"
        "<table class='dynamo-table'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{body_html}</tbody>"
        "</table>"
        "</div>"
    )


def get_empty_tables():
    """Get empty table HTML for all three table types."""
    return (
        build_table_html(PREFILL_TABLE_HEADERS, []),
        build_table_html(DECODE_TABLE_HEADERS, []),
        build_table_html(COST_TABLE_HEADERS, []),
    )


def prepare_prefill_table_data(prefill_results):
    """
    Prepare table data for prefill performance.

    Args:
        prefill_results: Tuple of (num_gpus_list, ttft_list, thpt_per_gpu_list)

    Returns:
        List of rows for the table
    """
    num_gpus_list, ttft_list, thpt_per_gpu_list = prefill_results
    return [
        [num_gpus, round(ttft, 3), round(thpt, 3)]
        for num_gpus, ttft, thpt in zip(num_gpus_list, ttft_list, thpt_per_gpu_list)
    ]


def prepare_decode_table_data(decode_results):
    """
    Prepare table data for decode performance.

    Args:
        decode_results: List of tuples (num_gpus, itl_list, thpt_list)

    Returns:
        List of rows for the table
    """
    table_data = []
    for num_gpus, itl_list, thpt_list in decode_results:
        for itl, thpt in zip(itl_list, thpt_list):
            table_data.append([num_gpus, round(itl, 3), round(thpt, 3)])
    return table_data


def prepare_cost_table_data(
    isl, osl, prefill_results, decode_results, gpu_cost_per_hour
):
    """
    Prepare table data for cost analysis.

    Args:
        isl: Input sequence length
        osl: Output sequence length
        prefill_results: Tuple of (num_gpus, ttft, thpt_per_gpu) for prefill
        decode_results: List of tuples (num_gpus, itl_list, thpt_per_gpu_list) for decode
        gpu_cost_per_hour: Cost per GPU per hour in dollars

    Returns:
        List of rows for the table
    """
    # Compute Pareto fronts
    p_ttft, p_thpt = compute_parato(prefill_results[1], prefill_results[2])

    _d_itl, _d_thpt = [], []
    for _d_result in decode_results:
        _d_itl.extend(_d_result[1])
        _d_thpt.extend(_d_result[2])
    d_itl, d_thpt = compute_parato(_d_itl, _d_thpt)

    # Convert to numpy arrays
    p_ttft = np.array(p_ttft)
    p_thpt = np.array(p_thpt)
    d_itl = np.array(d_itl)
    d_thpt = np.array(d_thpt)

    # Calculate cost data
    table_data = []
    for _p_ttft, _p_thpt in zip(p_ttft, p_thpt):
        prefill_cost = isl * 1000 / _p_thpt * gpu_cost_per_hour / 3600
        tokens_per_user_array = 1000 / d_itl
        cost_array = osl * 1000 / d_thpt * gpu_cost_per_hour / 3600 + prefill_cost

        for i in range(len(d_itl)):
            table_data.append(
                [
                    round(float(_p_ttft), 3),
                    round(float(_p_thpt), 3),
                    round(float(d_itl[i]), 3),
                    round(float(d_thpt[i]), 3),
                    round(float(tokens_per_user_array[i]), 3),
                    round(float(cost_array[i]), 3),
                ]
            )

    return table_data


def build_all_tables(prefill_results, decode_results, isl, osl, gpu_cost_per_hour):
    """
    Build all three table HTMLs from profiling results.

    Args:
        prefill_results: Prefill profiling results
        decode_results: Decode profiling results
        isl: Input sequence length
        osl: Output sequence length
        gpu_cost_per_hour: Cost per GPU per hour

    Returns:
        Tuple of (prefill_table_html, decode_table_html, cost_table_html)
    """
    prefill_data = prepare_prefill_table_data(prefill_results)
    decode_data = prepare_decode_table_data(decode_results)
    cost_data = prepare_cost_table_data(
        isl, osl, prefill_results, decode_results, gpu_cost_per_hour
    )

    return (
        build_table_html(PREFILL_TABLE_HEADERS, prefill_data),
        build_table_html(DECODE_TABLE_HEADERS, decode_data),
        build_table_html(COST_TABLE_HEADERS, cost_data),
    )
