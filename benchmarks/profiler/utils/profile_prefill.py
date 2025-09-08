# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

import numpy as np

from benchmarks.profiler.utils.genai_perf import benchmark_prefill
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


def profile_prefill(
    work_dir,
    model_name,
    tokenizer,
    url,
    num_gpus,
    max_context_length,
    interpolation_granularity,
):
    prefill_isl = []
    prefill_ttft = []
    prefill_thpt_per_gpu = []
    for isl in range(
        100,
        max_context_length,
        (max_context_length - 100) // interpolation_granularity,
    ):
        # run genai-perf
        genai_perf_artifact_dir = f"{work_dir}/gap_isl{isl}"
        gap_result = benchmark_prefill(
            isl,
            genai_perf_artifact_dir,
            model_name,
            tokenizer,
            base_url=url,
        )
        if gap_result is not None:
            ttft = gap_result["time_to_first_token"]["avg"]
            prefill_isl.append(isl)
            prefill_ttft.append(ttft)
            prefill_thpt_per_gpu.append(isl / ttft / num_gpus * 1000)

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
