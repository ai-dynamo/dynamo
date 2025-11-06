# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Orchestration logic for generating performance plots.

This module contains the main pipeline that coordinates profiling,
plot generation, and table building.
"""

from benchmarks.profiler.webapp.core.profiling import (
    format_status_message,
    generate_gpu_configurations,
    initialize_ai_configurator,
    profile_decode_performance,
    profile_prefill_performance,
    validate_inputs,
)
from benchmarks.profiler.webapp.ui.plots import (
    plot_cost_sla_interactive,
    plot_decode_performance_interactive,
    plot_prefill_performance_interactive,
)
from benchmarks.profiler.webapp.ui.tables import build_all_tables, get_empty_tables


def generate_plots(
    aic_model_name: str,
    backend: str,
    config_yaml: str,
    use_aic: bool,
    aic_backend: str,
    aic_backend_version: str,
    aic_system: str,
    min_num_gpus_per_engine: int,
    max_num_gpus_per_engine: int,
    num_gpus_per_node: int,
    gpu_cost_per_hour: float,
    isl: int,
    osl: int,
    max_context_length: int,
    ttft: float,
    itl: float,
):
    """
    Generate performance plots using AI Configurator estimation.

    This function profiles LLM inference performance by:
    1. Estimating prefill performance (TTFT) across different GPU counts
    2. Estimating decode performance (ITL) at various concurrency levels
    3. Computing cost-vs-SLA tradeoffs based on GPU pricing

    Args:
        aic_model_name: Model name for AI Configurator (e.g., "QWEN3_32B")
        backend: Inference backend (vllm, sglang, trtllm) - for reference only
        config_yaml: YAML configuration string from UI (reserved for future use)
        use_aic: Whether to use AI Configurator (must be True for webapp)
        aic_backend: Backend for AI Configurator estimation
        aic_backend_version: Version of the backend
        aic_system: GPU system (e.g., "H200_SXM")
        min_num_gpus_per_engine: Minimum TP size to profile
        max_num_gpus_per_engine: Maximum TP size to profile
        num_gpus_per_node: GPUs per node (for MoE models, unused for dense)
        gpu_cost_per_hour: Cost per GPU per hour in dollars
        isl: Input sequence length
        osl: Output sequence length
        max_context_length: Maximum context length (currently unused)
        ttft: Target TTFT in milliseconds (for visualization)
        itl: Target ITL in milliseconds (for visualization)

    Returns:
        Tuple of (prefill_plot, decode_plot, cost_plot, status_message,
                  prefill_table_html, decode_table_html, cost_table_html)
    """
    empty_prefill_html, empty_decode_html, empty_cost_html = get_empty_tables()

    try:
        # Validate inputs
        is_valid, error_msg = validate_inputs(
            use_aic, aic_model_name, aic_system, aic_backend_version
        )
        if not is_valid:
            return (
                None,
                None,
                None,
                error_msg,
                empty_prefill_html,
                empty_decode_html,
                empty_cost_html,
            )

        # Initialize AI Configurator
        ai_configurator = initialize_ai_configurator(
            aic_model_name, aic_system, aic_backend, aic_backend_version
        )

        # Generate GPU configurations to profile
        profile_num_gpus = generate_gpu_configurations(
            min_num_gpus_per_engine, max_num_gpus_per_engine
        )

        if not profile_num_gpus:
            return (
                None,
                None,
                None,
                "❌ No valid GPU configurations to profile",
                empty_prefill_html,
                empty_decode_html,
                empty_cost_html,
            )

        # Profile prefill performance
        prefill_results = profile_prefill_performance(
            ai_configurator, profile_num_gpus, isl
        )

        if not prefill_results[0]:
            return (
                None,
                None,
                None,
                "❌ Failed to generate prefill results",
                empty_prefill_html,
                empty_decode_html,
                empty_cost_html,
            )

        # Profile decode performance
        decode_results = profile_decode_performance(
            ai_configurator, profile_num_gpus, isl, osl
        )

        if not decode_results:
            return (
                None,
                None,
                None,
                "❌ Failed to generate decode results",
                empty_prefill_html,
                empty_decode_html,
                empty_cost_html,
            )

        # Generate interactive plots
        prefill_plot = plot_prefill_performance_interactive(prefill_results, ttft)
        decode_plot = plot_decode_performance_interactive(decode_results, itl)
        cost_plot = plot_cost_sla_interactive(
            isl, osl, prefill_results, decode_results, gpu_cost_per_hour
        )

        # Generate success status message
        status_msg = format_status_message(
            profile_num_gpus, prefill_results, gpu_cost_per_hour
        )

        # Build all tables
        prefill_table_html, decode_table_html, cost_table_html = build_all_tables(
            prefill_results, decode_results, isl, osl, gpu_cost_per_hour
        )

        return (
            prefill_plot,
            decode_plot,
            cost_plot,
            status_msg,
            prefill_table_html,
            decode_table_html,
            cost_table_html,
        )

    except Exception as e:
        import traceback

        error_msg = f"❌ Error generating plots:\n{str(e)}\n\n{traceback.format_exc()}"
        return (
            None,
            None,
            None,
            error_msg,
            empty_prefill_html,
            empty_decode_html,
            empty_cost_html,
        )
