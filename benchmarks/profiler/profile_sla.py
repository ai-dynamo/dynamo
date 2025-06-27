# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import math
import os
import subprocess

import numpy as np
import yaml
from utils.config import CONFIG_MODIFIERS
from utils.defaults import DECODE_NUM_REQUESTS_RANGE
from utils.genai_perf import benchmark_decode, benchmark_prefill
from utils.plot import (
    plot_decode_3d_surface,
    plot_decode_performance,
    plot_prefill_interpolation,
    plot_prefill_performance,
)
from utils.utils import (
    get_available_gpu_count,
    get_dynamo_serve_cmd,
    shutdown_deployment,
    wait_for_server_ready,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm_v0",
        choices=["vllm_v0", "vllm_v1"],
        help="backend type (currently only vllm is supported)",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the dynamo config file"
    )
    parser.add_argument(
        "--example-dir",
        type=str,
        default=None,
        help="path to the example directory, if not provided, will try to infer from config file location",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiling_results",
        help="Path to the output results directory",
    )
    parser.add_argument(
        "--isl", type=int, default=3000, help="target input sequence length"
    )
    parser.add_argument(
        "--osl", type=int, default=500, help="target output sequence length"
    )
    parser.add_argument(
        "--ttft", type=int, default=50, help="target Time To First Token in ms"
    )
    parser.add_argument(
        "--itl", type=int, default=10, help="target Inter Token Latency in ms"
    )
    # below are arguments used for interpolating TTFT and ITL under different ISL/OSL
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=16384,
        help="maximum context length supported by the served model",
    )
    parser.add_argument(
        "--prefill-interpolation-granularity",
        type=int,
        default=16,
        help="how many samples to benchmark to interpolate TTFT under different ISL",
    )
    parser.add_argument(
        "--decode-interpolation-granularity",
        type=int,
        default=6,
        help="how many samples to benchmark to interpolate ITL under different active kv cache size and decode context length",
    )
    args = parser.parse_args()

    config_modifier = CONFIG_MODIFIERS[args.backend]

    if args.example_dir is None:
        logger.info(
            "Example directory not provided, inferring from config file location..."
        )
        try:
            args.example_dir = os.path.dirname(os.path.dirname(args.config))
        except Exception:
            logger.error(
                "Failed to infer example directory, please provide explicitly using --example-dir <path-to-example-dir>"
            )
            exit(1)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Get the number of available GPUs
    available_gpus = get_available_gpu_count()

    profile_tp_size = [2**i for i in range(int(math.log2(available_gpus)) + 1)]
    logger.info(f"Profiling TP sizes: {profile_tp_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = config_modifier.get_model_name(config)
    port = config_modifier.get_port(config)

    # first profile prefill
    prefill_tp_size = []
    prefill_ttft = []
    prefill_thpt_per_gpu = []
    logger.info("Profiling prefill...")
    prefill_config = config_modifier.convert_config(config, "prefill")
    for tp_size in profile_tp_size:
        logger.info(f"Profiling prefill with TP size {tp_size}...")
        prefill_config = config_modifier.set_config_tp_size(prefill_config, tp_size)
        logger.info(f"Dynamo config: {prefill_config}")

        work_dir = f"{args.output_dir}/prefill_tp{tp_size}"
        os.makedirs(work_dir, exist_ok=True)

        prefill_config_fn = f"{work_dir}/config.yaml"
        dynamo_log_fn = f"{work_dir}/dynamo.log"
        with open(prefill_config_fn, "w") as f:
            yaml.dump(prefill_config, f)

        # Start the dynamo serve process
        logger.info(f"Starting dynamo serve with TP size {tp_size}...")
        dynamo_serve_cmd = get_dynamo_serve_cmd(prefill_config_fn)
        with open(dynamo_log_fn, "w") as dynamo_log_f:
            dynamo_process = subprocess.Popen(
                dynamo_serve_cmd,
                stdout=dynamo_log_f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=args.example_dir,
                preexec_fn=os.setsid,  # Use process group for clean termination
            )

        if not wait_for_server_ready(model_name, port):
            logger.error(f"Server did not become ready, skip profiling tp={tp_size}")
            break

        # run genai-perf
        genai_perf_artifact_dir = f"{work_dir}/gap_isl{args.isl}"
        gap_result = benchmark_prefill(
            args.isl, genai_perf_artifact_dir, model_name, port
        )
        if gap_result is not None:
            ttft = gap_result["time_to_first_token"]["avg"]
            prefill_tp_size.append(tp_size)
            prefill_ttft.append(ttft)
            prefill_thpt_per_gpu.append(args.isl / ttft / tp_size * 1000)

        shutdown_deployment(dynamo_process)

    # Plot the results as a 2D scatter plot
    if prefill_tp_size and prefill_ttft and prefill_thpt_per_gpu:
        plot_prefill_performance(
            prefill_tp_size,
            prefill_ttft,
            prefill_thpt_per_gpu,
            args.ttft,
            args.output_dir,
        )

    # then profile decode
    decode_tp_size = []
    decode_itl = []
    decode_thpt_per_gpu = []
    decode_concurrency = []
    decode_kv_cache_size = []
    decode_results = []  # Store partial results for plotting later
    logger.info("Profiling decode...")
    decode_config = config_modifier.convert_config(config, "decode")
    for tp_size in profile_tp_size:
        logger.info(f"Profiling decode with TP size {tp_size}...")
        decode_config = config_modifier.set_config_tp_size(decode_config, tp_size)
        logger.info(f"Dynamo config: {decode_config}")

        work_dir = f"{args.output_dir}/decode_tp{tp_size}"
        os.makedirs(work_dir, exist_ok=True)

        decode_config_fn = f"{work_dir}/config.yaml"
        dynamo_log_fn = f"{work_dir}/dynamo.log"
        with open(decode_config_fn, "w") as f:
            yaml.dump(decode_config, f)

        # Start the dynamo serve process
        logger.info(f"Starting dynamo serve with TP size {tp_size}...")
        dynamo_serve_cmd = get_dynamo_serve_cmd(decode_config_fn)
        with open(dynamo_log_fn, "w") as dynamo_log_f:
            dynamo_process = subprocess.Popen(
                dynamo_serve_cmd,
                stdout=dynamo_log_f,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=args.example_dir,
                preexec_fn=os.setsid,  # Use process group for clean termination
            )

        if not wait_for_server_ready(model_name, port):
            logger.error(f"Server did not become ready, skip profiling tp={tp_size}")
            break

        max_kv_tokens = config_modifier.get_kv_cache_size_from_dynamo_log(dynamo_log_fn)
        max_concurrency = max_kv_tokens // (args.isl + args.osl)
        sweep_num_request = [
            num for num in DECODE_NUM_REQUESTS_RANGE if num < max_concurrency
        ]
        logger.info(
            f"Sweeping num_request range based on maximum number of kv tokens: {sweep_num_request}"
        )

        engine_decode_itl = []
        engine_decode_thpt_per_gpu = []
        for num_request in sweep_num_request:
            genai_perf_artifact_dir = f"{work_dir}/gap_request{num_request}_isl{args.isl}_osl{args.osl}_n{num_request}"
            gap_result = benchmark_decode(
                args.isl,
                args.osl,
                num_request,
                genai_perf_artifact_dir,
                model_name,
                port,
            )
            if gap_result is not None:
                itl = gap_result["inter_token_latency"]["avg"]
                thpt_per_gpu = gap_result["output_token_throughput"]["avg"] / tp_size
                engine_decode_itl.append(itl)
                engine_decode_thpt_per_gpu.append(thpt_per_gpu)
                decode_tp_size.append(tp_size)
                decode_itl.append(itl)
                decode_thpt_per_gpu.append(thpt_per_gpu)
                decode_concurrency.append(num_request)
                decode_kv_cache_size.append(max_kv_tokens)

        shutdown_deployment(dynamo_process)

        # Store partial results for plotting later
        decode_results.append((tp_size, engine_decode_itl, engine_decode_thpt_per_gpu))

    # Plot all decode results after profiling is complete
    if decode_results:
        plot_decode_performance(decode_results, args.itl, args.output_dir)

    logger.info("Analyzing results and generate recommendations...")
    # select best tp size for prefill
    if min(prefill_ttft) > args.ttft:
        logger.info(
            "No TP size satisfies the TTFT requirement, please try a smaller model or a more powerful GPU SKU"
        )
        selected_prefill_idx = int(np.argmin(np.array(prefill_ttft)))
    else:
        valid_indices = [i for i, ttft in enumerate(prefill_ttft) if ttft <= args.ttft]
        # Among valid TP sizes, select the one with highest throughput per GPU
        valid_thpts = [prefill_thpt_per_gpu[i] for i in valid_indices]
        max_thpt_idx = valid_indices[int(np.argmax(valid_thpts))]
        selected_prefill_idx = max_thpt_idx
    logger.info(
        f"Suggested prefill TP:{prefill_tp_size[selected_prefill_idx]} (TTFT {prefill_ttft[selected_prefill_idx]:.2f} ms, throughput {prefill_thpt_per_gpu[selected_prefill_idx]:.2f} tokens/s/GPU)"
    )

    # scale up if estimated TTFT is 120% of target TTFT
    prefill_queue_size_upper_bound = max(
        0.1, args.ttft * 1.2 / prefill_ttft[selected_prefill_idx] - 1
    )
    # scale down if estimated TTFT is 80% of target TTFT
    prefill_queue_size_lower_bound = max(
        0.1, args.ttft * 0.8 / prefill_ttft[selected_prefill_idx] - 1
    )
    logger.info(
        f"Suggested planner upper/lower bound for prefill queue size: {prefill_queue_size_upper_bound:.2f}/{prefill_queue_size_lower_bound:.2f}"
    )

    # select best tp size for decode
    if min(decode_itl) > args.itl:
        logger.info(
            "No TP size satisfies the ITL requirement, please try a smaller model or a more powerful GPU SKU"
        )
        selected_decode_idx = int(np.argmin(np.array(decode_itl)))
    else:
        valid_indices = [i for i, itl in enumerate(decode_itl) if itl <= args.itl]
        # Among valid TP sizes, select the one with highest throughput per GPU
        valid_thpts = [decode_thpt_per_gpu[i] for i in valid_indices]
        max_thpt_idx = valid_indices[int(np.argmax(valid_thpts))]
        selected_decode_idx = max_thpt_idx
    logger.info(
        f"Suggested decode TP:{decode_tp_size[selected_decode_idx]} (ITL {decode_itl[selected_decode_idx]:.2f} ms, throughput {decode_thpt_per_gpu[selected_decode_idx]:.2f} tokens/s/GPU)"
    )

    # calculate kv cache utlization for the selected TP and concurrency
    selected_decode_kv_cache_utilization = (
        decode_concurrency[selected_decode_idx]
        * (args.isl + args.osl / 2)
        / decode_kv_cache_size[selected_decode_idx]
    )
    # set a +- 20% range for the kv cache utilization
    logger.info(
        f"Suggested planner upper/lower bound for decode kv cache utilization: {min(1, selected_decode_kv_cache_utilization + 0.2):.2f}/{max(0.1, selected_decode_kv_cache_utilization - 0.2):.2f}"
    )

    # interpolate ISL - TTFT with best prefill TP
    best_prefill_tp = prefill_tp_size[selected_prefill_idx]
    prefill_isl = []
    prefill_ttft = []
    prefill_thpt_per_gpu = []
    logger.info(
        f"Profiling prefill under best TP {best_prefill_tp} with different ISL..."
    )
    prefill_config = config_modifier.convert_config(config, "prefill")
    prefill_config = config_modifier.set_config_tp_size(prefill_config, tp_size)
    logger.info(f"Dynamo config: {prefill_config}")

    work_dir = f"{args.output_dir}/selected_prefill_interpolation"
    os.makedirs(work_dir, exist_ok=True)

    prefill_config_fn = f"{work_dir}/config.yaml"

    dynamo_log_fn = f"{work_dir}/dynamo.log"
    with open(prefill_config_fn, "w") as f:
        yaml.dump(prefill_config, f)

    # Start the dynamo serve process
    logger.info(f"Starting dynamo serve with TP size {tp_size}...")
    dynamo_serve_cmd = get_dynamo_serve_cmd(prefill_config_fn)
    with open(dynamo_log_fn, "w") as dynamo_log_f:
        dynamo_process = subprocess.Popen(
            dynamo_serve_cmd,
            stdout=dynamo_log_f,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=args.example_dir,
            preexec_fn=os.setsid,  # Use process group for clean termination
        )

    if not wait_for_server_ready(model_name, port):
        logger.error(f"Server did not become ready, skip profiling tp={tp_size}")
    else:
        for isl in range(
            100,
            args.max_context_length,
            (args.max_context_length - 100) // args.prefill_interpolation_granularity,
        ):
            # run genai-perf
            genai_perf_artifact_dir = f"{work_dir}/gap_isl{isl}"
            gap_result = benchmark_prefill(
                isl, genai_perf_artifact_dir, model_name, port
            )
            if gap_result is not None:
                ttft = gap_result["time_to_first_token"]["avg"]
                prefill_isl.append(isl)
                prefill_ttft.append(ttft)
                prefill_thpt_per_gpu.append(isl / ttft / best_prefill_tp * 1000)

    shutdown_deployment(dynamo_process)

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

    # interpolate ITL - Active_KV_Cache - Decode_Context_Length with best decode TP
    x_kv_usage = []
    y_context_length = []
    z_itl = []
    z_thpt_per_gpu = []
    best_decode_tp = decode_tp_size[selected_decode_idx]
    logger.info(f"Profiling decode with TP size {best_decode_tp}...")
    decode_config = config_modifier.set_config_tp_size(decode_config, best_decode_tp)
    logger.info(f"Dynamo config: {decode_config}")

    work_dir = f"{args.output_dir}/selected_decode_interpolation"
    os.makedirs(work_dir, exist_ok=True)

    decode_config_fn = f"{work_dir}/config.yaml"
    dynamo_log_fn = f"{work_dir}/dynamo.log"
    with open(decode_config_fn, "w") as f:
        yaml.dump(decode_config, f)

    # Start the dynamo serve process
    logger.info(f"Starting dynamo serve with TP size {tp_size}...")
    dynamo_serve_cmd = get_dynamo_serve_cmd(decode_config_fn)
    with open(dynamo_log_fn, "w") as dynamo_log_f:
        dynamo_process = subprocess.Popen(
            dynamo_serve_cmd,
            stdout=dynamo_log_f,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=args.example_dir,
            preexec_fn=os.setsid,  # Use process group for clean termination
        )

    if not wait_for_server_ready(model_name, port):
        logger.error(f"Server did not become ready, skip profiling tp={tp_size}")
    else:
        max_kv_tokens = config_modifier.get_kv_cache_size_from_dynamo_log(dynamo_log_fn)

        osl = 500  # not too large to reduce ITL variance, not too small to have stable measurement
        for isl in range(
            100,
            args.max_context_length - osl,
            (args.max_context_length - osl) // args.decode_interpolation_granularity,
        ):
            max_concurrency = max_kv_tokens // (isl + osl)
            sweep_num_request = list(
                range(
                    1,
                    max_concurrency,
                    max_concurrency // args.decode_interpolation_granularity,
                )
            )
            for num_request in sweep_num_request:
                genai_perf_artifact_dir = (
                    f"{work_dir}/gap_isl{isl}_osl{osl}_n{num_request}"
                )
                gap_result = benchmark_decode(
                    isl, osl, num_request, genai_perf_artifact_dir, model_name, port
                )
                if gap_result is not None:
                    itl = gap_result["inter_token_latency"]["avg"]
                    x_kv_usage.append((isl + osl / 2) * num_request / max_kv_tokens)
                    y_context_length.append(isl + osl / 2)
                    z_itl.append(itl)
                    z_thpt_per_gpu.append(
                        gap_result["output_token_throughput"]["avg"] / tp_size
                    )

        shutdown_deployment(dynamo_process)

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
            x_kv_usage, y_context_length, z_itl, best_decode_tp, work_dir
        )
