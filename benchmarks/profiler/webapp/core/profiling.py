# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Performance profiling logic for the Dynamo SLA Profiler webapp.

This module handles the actual performance estimation using AI Configurator,
including prefill and decode performance profiling.
"""

import math

from benchmarks.profiler.utils.estimate_perf import AIConfiguratorPerfEstimator
from benchmarks.profiler.utils.profile_decode import get_num_request_range
from benchmarks.profiler.webapp.core.constants import (
    DEFAULT_DECODE_INTERPOLATION_GRANULARITY,
)


def validate_inputs(use_aic, aic_model_name, aic_system, aic_backend_version):
    """
    Validate AI Configurator inputs.

    Args:
        use_aic: Whether AI Configurator is enabled
        aic_model_name: Model name for AI Configurator
        aic_system: GPU system name
        aic_backend_version: Backend version

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not use_aic:
        return False, "❌ Web UI requires AI Configurator mode"

    if not aic_model_name or not aic_system or not aic_backend_version:
        return False, "❌ Missing required AI Configurator parameters"

    return True, None


def initialize_ai_configurator(
    aic_model_name, aic_system, aic_backend, aic_backend_version
):
    """
    Initialize AI Configurator Performance Estimator.

    Args:
        aic_model_name: Model name for AI Configurator
        aic_system: GPU system (e.g., "H200_SXM")
        aic_backend: Backend for AI Configurator estimation
        aic_backend_version: Version of the backend

    Returns:
        AIConfiguratorPerfEstimator instance
    """
    return AIConfiguratorPerfEstimator(
        aic_model_name,
        aic_system.lower(),
        aic_backend,
        aic_backend_version,
    )


def generate_gpu_configurations(min_num_gpus, max_num_gpus):
    """
    Generate GPU counts to profile (powers of 2 for dense models).

    Args:
        min_num_gpus: Minimum number of GPUs
        max_num_gpus: Maximum number of GPUs

    Returns:
        List of GPU counts to profile
    """
    profile_num_gpus = [
        2**i
        for i in range(int(math.log2(max_num_gpus)) + 1)
        if min_num_gpus <= 2**i <= max_num_gpus
    ]
    return profile_num_gpus


def profile_prefill_performance(ai_configurator, profile_num_gpus, isl):
    """
    Profile prefill performance across different GPU counts.

    Args:
        ai_configurator: AIConfiguratorPerfEstimator instance
        profile_num_gpus: List of GPU counts to profile
        isl: Input sequence length

    Returns:
        Tuple of (num_gpus_list, ttft_list, thpt_per_gpu_list)
    """
    prefill_num_gpus = []
    prefill_ttft = []
    prefill_thpt_per_gpu = []

    for num_gpus in profile_num_gpus:
        # Estimate prefill performance using AI Configurator
        perf_dict = ai_configurator.estimate_prefill_perf(
            isl,
            tp_size=num_gpus,
        )
        ttft_val = perf_dict["context_latency"]
        # Calculate throughput: tokens/second/GPU
        thpt_val = isl / ttft_val * 1000 / num_gpus

        prefill_num_gpus.append(num_gpus)
        prefill_ttft.append(ttft_val)
        prefill_thpt_per_gpu.append(thpt_val)

    return (prefill_num_gpus, prefill_ttft, prefill_thpt_per_gpu)


def profile_decode_performance(
    ai_configurator,
    profile_num_gpus,
    isl,
    osl,
    decode_interpolation_granularity=DEFAULT_DECODE_INTERPOLATION_GRANULARITY,
):
    """
    Profile decode performance at various concurrency levels.

    Args:
        ai_configurator: AIConfiguratorPerfEstimator instance
        profile_num_gpus: List of GPU counts to profile
        isl: Input sequence length
        osl: Output sequence length
        decode_interpolation_granularity: Granularity for decode interpolation

    Returns:
        List of tuples (num_gpus, itl_list, thpt_per_gpu_list)
    """
    decode_results = []
    # For dense models (not MoE), attention_dp_size = 1
    attention_dp_size = 1

    for num_gpus in profile_num_gpus:
        # Get maximum batch size for this configuration
        max_concurrency = ai_configurator.get_max_batch_size(isl, osl, tp_size=num_gpus)

        # Determine request sweep range
        sweep_num_request = get_num_request_range(
            attention_dp_size,
            max_concurrency,
            decode_interpolation_granularity,
        )

        engine_decode_itl = []
        engine_decode_thpt_per_gpu = []

        for num_request in sweep_num_request:
            # Estimate decode performance using AI Configurator
            perf_dict = ai_configurator.estimate_perf(
                isl,
                osl,
                num_request,
                mode="decode",
                tp_size=num_gpus,
            )

            itl_val = perf_dict["tpot"]
            thpt_val = perf_dict["tokens/s/gpu"]

            engine_decode_itl.append(itl_val)
            engine_decode_thpt_per_gpu.append(thpt_val)

        # Store results for this GPU configuration
        if engine_decode_itl:
            decode_results.append(
                (num_gpus, engine_decode_itl, engine_decode_thpt_per_gpu)
            )

    return decode_results


def format_status_message(profile_num_gpus, prefill_results, gpu_cost_per_hour):
    """
    Format success status message with profiling summary.

    Args:
        profile_num_gpus: List of GPU counts profiled
        prefill_results: Prefill profiling results
        gpu_cost_per_hour: Cost per GPU per hour

    Returns:
        Formatted status message string
    """
    _, prefill_ttft, _ = prefill_results
    prefill_num_gpus, _, _ = prefill_results

    best_prefill_idx = prefill_ttft.index(min(prefill_ttft))
    return (
        f"✅ Plots generated successfully!\n"
        f"📊 Profiled {len(profile_num_gpus)} GPU configurations: {profile_num_gpus}\n"
        f"⚡ Best prefill: {min(prefill_ttft):.1f}ms TTFT at {prefill_num_gpus[best_prefill_idx]} GPUs\n"
        f"💰 GPU Cost: ${gpu_cost_per_hour:.2f}/hour"
    )
