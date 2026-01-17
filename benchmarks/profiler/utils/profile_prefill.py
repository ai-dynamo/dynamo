# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Optional

import numpy as np

from benchmarks.profiler.utils.aiperf import get_prefill_ttft
from benchmarks.profiler.utils.estimate_perf import AIConfiguratorPerfEstimator
from benchmarks.profiler.utils.plot import plot_prefill_interpolation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def _profile_prefill_helper(
    work_dir,
    num_gpus,
    max_context_length,
    interpolation_granularity,
    get_ttft: Callable[[int], Optional[float]],
    attention_dp_size: int = 1,
):
    prefill_isl = []
    prefill_ttft = []
    prefill_thpt_per_gpu = []
    max_context_length -= 512  # leave some room for chat template and system prompt
    if max_context_length <= 100:
        error_message = (
            f"max_context_length {max_context_length} is too small to profile prefill"
        )
        logger.error(error_message)
        raise ValueError(error_message)
    for isl in range(
        100,
        max_context_length,
        (max_context_length - 100) // interpolation_granularity,
    ):
        ttft = get_ttft(isl)
        if ttft is not None:
            prefill_isl.append(isl)
            prefill_ttft.append(ttft)
            prefill_thpt_per_gpu.append(
                isl / ttft / num_gpus * 1000 * attention_dp_size
            )

    # Interpolate prefill_ttft vs prefill_isl with quadratic function (y=ax^2+bx+c)
    if len(prefill_isl) > 2:
        logger.info("Interpolating prefill TTFT and throughput vs ISL...")

        # Convert to numpy arrays for easier manipulation
        prefill_isl_np = np.array(prefill_isl)
        prefill_ttft_np = np.array(prefill_ttft)
        prefill_thpt_per_gpu_np = np.array(prefill_thpt_per_gpu)

        save_path = f"{work_dir}/raw_data.npz"
        np.savez(
            save_path,
            prefill_isl=prefill_isl_np,
            prefill_ttft=prefill_ttft_np,
            prefill_thpt_per_gpu=prefill_thpt_per_gpu_np,
        )

        # Call the plotting function
        plot_prefill_interpolation(
            prefill_isl_np, prefill_ttft_np, prefill_thpt_per_gpu_np, work_dir
        )
    else:
        logger.warning(
            "Not enough data points to perform interpolation (need at least 3 points)"
        )

    return


def profile_prefill(
    work_dir,
    model_name,
    tokenizer,
    url,
    num_gpus,
    max_context_length,
    interpolation_granularity,
    attention_dp_size: int = 1,
):
    def get_ttft(isl):
        ai_perf_artifact_dir = f"{work_dir}/aiperf_isl{isl}"
        return get_prefill_ttft(
            isl,
            ai_perf_artifact_dir,
            model_name,
            tokenizer,
            base_url=url,
            attention_dp_size=attention_dp_size,
        )

    return _profile_prefill_helper(
        work_dir,
        num_gpus,
        max_context_length,
        interpolation_granularity,
        get_ttft,
        attention_dp_size=attention_dp_size,
    )


def profile_prefill_aiconfigurator(
    work_dir,
    num_gpus,
    max_context_length,
    interpolation_granularity,
    ai_configurator_perf_estimator: AIConfiguratorPerfEstimator,
    **model_config_kwargs,
):
    def get_ttft(isl):
        perf_dict = ai_configurator_perf_estimator.estimate_prefill_perf(
            isl,
            **model_config_kwargs,
        )

        ttft = perf_dict["context_latency"]
        logger.info(f"Estimated prefill TTFT: {ttft:.2f}ms")
        return ttft

    return _profile_prefill_helper(
        work_dir,
        num_gpus,
        max_context_length,
        interpolation_granularity,
        get_ttft,
    )
