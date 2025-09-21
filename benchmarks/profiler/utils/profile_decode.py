# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Callable, Optional, Tuple

import numpy as np

from benchmarks.profiler.utils.estimate_perf import AIConfiguratorPerfEstimator
from benchmarks.profiler.utils.genai_perf import benchmark_decode
from benchmarks.profiler.utils.plot import plot_decode_3d_surface

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def _profile_decode_helper(
    work_dir,
    num_gpus,
    max_kv_tokens,
    max_context_length,
    interpolation_granularity,
    get_itl_and_thpt_per_gpu: Callable[
        [int, int, int], Tuple[Optional[float], Optional[float]]
    ],
):
    """interpolate ITL - Active_KV_Cache - Decode_Context_Length"""
    x_kv_usage = []
    y_context_length = []
    z_itl = []
    z_thpt_per_gpu = []

    osl = 500  # not too large to reduce ITL variance, not too small to have stable measurement

    for isl in range(
        100,
        max_context_length - osl,
        (max_context_length - osl) // interpolation_granularity,
    ):
        max_concurrency = max_kv_tokens // (isl + osl)
        if max_concurrency == 0:
            logger.warning(
                f"max_kv_tokens {max_kv_tokens} is too small for"
                f" isl {isl} + osl {osl}, skipping."
            )
            break
        elif max_concurrency < interpolation_granularity:
            logger.warning(
                f"max_concurrency {max_concurrency} is too small for"
                f" interpolation granularity {interpolation_granularity}."
                f" max_kv_tokens {max_kv_tokens}, isl {isl}, osl {osl}"
            )
            sweep_num_request = range(1, max_concurrency + 1)
        else:
            sweep_num_request = range(
                1,
                max_concurrency,
                max_concurrency // interpolation_granularity,
            )
        for num_request in sweep_num_request:
            itl, thpt_per_gpu = get_itl_and_thpt_per_gpu(isl, osl, num_request)

            if itl is not None and thpt_per_gpu is not None:
                x_kv_usage.append((isl + osl / 2) * num_request / max_kv_tokens)
                y_context_length.append(isl + osl / 2)
                z_itl.append(itl)
                z_thpt_per_gpu.append(thpt_per_gpu)

    # Save the data points to a .npz file
    save_path = f"{work_dir}/raw_data.npz"
    np.savez(
        save_path,
        x_kv_usage=np.array(x_kv_usage),
        y_context_length=np.array(y_context_length),
        z_itl=np.array(z_itl),
        z_thpt_per_gpu=np.array(z_thpt_per_gpu),
        max_kv_tokens=np.array([max_kv_tokens]),
    )
    logger.info(f"Saved data points to {save_path}")

    # Plot 3D surface
    plot_decode_3d_surface(
        x_kv_usage, y_context_length, z_itl, z_thpt_per_gpu, work_dir
    )

    return


def profile_decode(
    work_dir,
    model_name,
    tokenizer,
    url,
    num_gpus,
    max_kv_tokens,
    max_context_length,
    interpolation_granularity,
):
    def get_itl_and_thpt_per_gpu(isl, osl, num_request):
        genai_perf_artifact_dir = f"{work_dir}/gap_isl{isl}_osl{osl}_n{num_request}"
        gap_result = benchmark_decode(
            isl,
            osl,
            num_request,
            genai_perf_artifact_dir,
            model_name,
            tokenizer,
            base_url=url,
        )
        if gap_result is not None:
            itl = gap_result["inter_token_latency"]["avg"]
            thpt_per_gpu = gap_result["output_token_throughput"]["avg"] / num_gpus
            return itl, thpt_per_gpu
        return None, None

    return _profile_decode_helper(
        work_dir,
        num_gpus,
        max_kv_tokens,
        max_context_length,
        interpolation_granularity,
        get_itl_and_thpt_per_gpu,
    )


def profile_decode_aiconfigurator(
    work_dir,
    num_gpus,
    max_kv_tokens,
    max_context_length,
    interpolation_granularity,
    ai_configurator_perf_estimator: AIConfiguratorPerfEstimator,
    **model_config_kwargs,
):
    def get_itl_and_thpt_per_gpu(isl, osl, num_request):
        perf_dict = ai_configurator_perf_estimator.estimate_perf(
            isl,
            osl,
            num_request,
            mode="decode",
            **model_config_kwargs,
        )
        return perf_dict["tpot"], perf_dict["tokens/s/gpu"]

    return _profile_decode_helper(
        work_dir,
        num_gpus,
        max_kv_tokens,
        max_context_length,
        interpolation_granularity,
        get_itl_and_thpt_per_gpu,
    )
