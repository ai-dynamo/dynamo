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
import json
import logging
import math
import os
import random
import signal
import subprocess
import time
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import requests
import yaml
from matplotlib import cm
from scipy.interpolate import griddata

DECODE_NUM_REQUESTS_RANGE = [
    1,
    5,
    10,
    25,
    50,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
]

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def get_dynamo_serve_cmd(config_file_path):
    return [
        "dynamo",
        "serve",
        "graphs.agg:Frontend",
        "-f",
        config_file_path,
    ]


def _get_common_genai_perf_cmd(
    artifact_dir,
    seed=100,
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    port=8000,
):
    return [
        "genai-perf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        model,
        "--service-kind",
        "openai",
        "--endpoint-type",
        "chat",
        "--endpoint",
        "/v1/chat/completions",
        "--streaming",
        "--url",
        f"http://localhost:{port}",
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        '{"nvext":{"ignore_eos":true}}',
        "--warmup-request-count",
        "3",
        "--artifact-dir",
        artifact_dir,
        "--random-seed",
        str(seed),
    ]


def get_prefill_genai_perf_cmd(
    isl,
    artifact_dir,
    seed=100,
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    osl=5,
    port=8000,
):
    return _get_common_genai_perf_cmd(
        artifact_dir,
        seed,
        model,
        port,
    ) + [
        "--synthetic-input-tokens-mean",
        str(isl),
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        "5",
        "--output-tokens-stddev",
        "0",
        "--extra-inputs",
        "max_tokens:5",
        "--extra-inputs",
        "min_tokens:5",
        "--concurrency",
        "1",
        "--request-count",
        "1",
    ]


def get_decode_genai_perf_cmd(
    isl,
    osl,
    artifact_dir,
    num_request,
    seed=100,
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    port=8000,
):
    return _get_common_genai_perf_cmd(
        artifact_dir,
        seed,
        model,
        port,
    ) + [
        "--synthetic-input-tokens-mean",
        str(isl),
        "--synthetic-input-tokens-stddev",
        "0",
        "--output-tokens-mean",
        str(osl),
        "--output-tokens-stddev",
        "0",
        "--extra-inputs",
        f"max_tokens:{osl}",
        "--extra-inputs",
        f"min_tokens:{osl}",
        "--concurrency",
        str(num_request),
        "--num-dataset-entries",
        str(num_request),
        "--request-count",
        str(num_request),
    ]


def convert_config(config: dict, target: Literal["prefill", "decode"]) -> dict:
    config = config.copy()

    # all profiles runs with a single prefill/decode worker, hence router doesn't matter
    if "Common" in config and "router" in config["Common"]:
        config["Common"]["router"] = "round-robin"
    else:
        config["Processor"]["router"] = "round-robin"

    # disable planner
    if "Planner" in config:
        config["Planner"]["no-operation"] = True

    if target == "prefill":
        if "PrefillWorker" in config:
            # make PrefillWorker into VllmWorker
            del config["VllmWorker"]
            config["VllmWorker"] = config["PrefillWorker"]
            del config["PrefillWorker"]

        # to profile prefill, we disable prefix caching
        config["VllmWorker"]["enable-prefix-caching"] = False
    elif target == "decode":
        if "PrefillWorker" in config:
            del config["PrefillWorker"]

        # to profile prefill, we enable prefix caching to pass the prefill stage
        config["VllmWorker"]["enable-prefix-caching"] = True

    # set num workers to 1
    config["VllmWorker"]["ServiceArgs"]["workers"] = 1

    # set PP to 1
    if (
        "pipeline-parallel-size" in config["VllmWorker"]
        and config["VllmWorker"]["pipeline-parallel-size"] > 1
    ):
        logger.warning("Currently we only support TP, setting PP to 1")
        config["VllmWorker"]["pipeline-parallel-size"] = 1

    # always local prefill
    config["VllmWorker"]["remote-prefill"] = False
    config["VllmWorker"]["conditional-disagg"] = False

    return config


def set_config_tp_size(config: dict, tp_size: int):
    config["VllmWorker"]["tensor-parallel-size"] = tp_size
    config["VllmWorker"]["ServiceArgs"]["resources"]["gpu"] = tp_size
    return config


def get_available_gpu_count():
    try:
        import pynvml

        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()

        if gpu_count > 0:
            logger.info(f"Detected {gpu_count} GPUs in the system:")
            for i in range(gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_memory_mb = memory.total / (1024 * 1024)
                free_memory_mb = memory.free / (1024 * 1024)
                logger.info(
                    f"  GPU {i}: {name}, Total Memory: {total_memory_mb:.2f} MB, Free Memory: {free_memory_mb:.2f} MB"
                )
        else:
            logger.warning("No GPUs detected with pynvml.")

        pynvml.nvmlShutdown()
        return gpu_count
    except ImportError:
        logger.error(
            "pynvml module not found. Please install it with 'pip install pynvml'"
        )
        return 0
    except pynvml.NVMLError as e:
        logger.error(f"NVML Error: {e}")
        return 0
    except Exception as e:
        logger.error(f"Error detecting GPUs: {e}")
        return 0


def get_model_name(config: dict) -> str:
    if "Common" in config and "served_model_name" in config["Common"]:
        return config["Common"]["served_model_name"]
    else:
        return config["Frontend"]["served_model_name"]


def get_port(config: dict) -> int:
    if "Common" in config and "port" in config["Common"]:
        return config["Common"]["port"]
    else:
        return config["Frontend"]["port"]


def shutdown_deployment(dynamo_process):
    os.killpg(os.getpgid(dynamo_process.pid), signal.SIGINT)
    dynamo_process.communicate()

    try:
        current_pid = os.getpid()
        ps_cmd = ["ps", "-ef"]
        ps_output = subprocess.check_output(ps_cmd, text=True)
        for line in ps_output.splitlines():
            if "python" in line.lower():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        pid = int(parts[1])
                        if pid != current_pid:  # Exclude current process
                            os.kill(pid, signal.SIGKILL)
                    except ValueError:
                        continue
    except Exception as e:
        logger.error(f"Error killing Python processes: {e}")
    time.sleep(5)


def wait_for_server_ready(model_name: str, port: int, timeout: int = 300):
    logger.info("Waiting for the server to be ready...")
    endpoint_url = f"http://localhost:{port}/v1/chat/completions"
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < timeout:
        try:
            # Send a simple request to check if the server is up
            response = requests.post(
                endpoint_url,
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1,
                },
                timeout=5,
            )
            if response.status_code != 200:
                logger.info(
                    f"Server returned status code {response.status_code}, waiting..."
                )
                time.sleep(5)
                continue
            logger.info(f"Server is ready after {time.time() - start_time:.2f} seconds")
            server_ready = True
            break

        except (requests.RequestException, ConnectionError) as e:
            logger.info(f"Server not ready yet: {e}")
        time.sleep(5)

    return server_ready


def get_kv_cache_size_from_dynamo_log(dynamo_log_fn: str) -> int:
    try:
        with open(dynamo_log_fn, "r") as f:
            for line in f:
                if "Maximum concurrency for" in line:
                    line = line.strip().split("Maximum concurrency for ")[1]
                    token_count = int(line.split(" tokens per request: ")[0])
                    concurrency = float(line.split(" tokens per request: ")[1][:-1])

                    logger.info(
                        f"Found KV cache info: {token_count} x {concurrency} = {int(token_count * concurrency)}"
                    )
                    return int(token_count * concurrency)
    except Exception as e:
        logger.warning(f"Failed to parse KV cache size from line: {line}. Error: {e}")
    return 0


def get_gap_result(artifact_dir: str) -> dict:
    with open(f"{artifact_dir}/profile_export_genai_perf.json", "r") as f:
        return json.load(f)


def benchmark_prefill(isl, genai_perf_artifact_dir, model_name, port):
    logger.info(f"Running genai-perf with isl {isl}")
    genai_perf_cmd = get_prefill_genai_perf_cmd(
        isl, genai_perf_artifact_dir, model=model_name, port=port
    )
    gap_process = subprocess.Popen(
        genai_perf_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = gap_process.communicate()
    if gap_process.returncode == 0:
        logger.info("Genai-perf profiling completed successfully")
        logger.info(stdout)
        gap_result = get_gap_result(genai_perf_artifact_dir)
        return gap_result
    else:
        logger.error(f"Genai-perf failed with error code: {gap_process.returncode}")
        logger.error(f"stderr: {stderr}")
        return None


def benchmark_decode(isl, osl, num_request, genai_perf_artifact_dir, model_name, port):
    logger.info(f"Profiling decode with num_request {num_request}...")

    # first warm-up the engine by pre-computing all prefill tokens
    # we use the same random seed to make sure the prompt is the same
    seed = random.randint(0, 1000000)
    genai_perf_cmd = get_decode_genai_perf_cmd(
        args.isl,
        args.osl,
        f"{genai_perf_artifact_dir}_warmup",
        num_request,
        seed=seed,
        model=model_name,
        port=port,
    )
    gap_process = subprocess.Popen(
        genai_perf_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    gap_process.communicate()
    # then send out the real requests, hopefully, this will skip all prefill computation
    genai_perf_cmd = get_decode_genai_perf_cmd(
        args.isl,
        args.osl,
        genai_perf_artifact_dir,
        num_request,
        seed=seed,
        model=model_name,
        port=port,
    )
    gap_process = subprocess.Popen(
        genai_perf_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = gap_process.communicate()
    if gap_process.returncode == 0:
        logger.info("Genai-perf profiling completed successfully")
        logger.info(stdout)
        gap_result = get_gap_result(genai_perf_artifact_dir)
        return gap_result
    else:
        logger.error(f"Genai-perf failed with error code: {gap_process.returncode}")
        logger.error(f"stderr: {stderr}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the dynamo config file"
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
        "--itl", type=int, default=5, help="target Inter Token Latency in ms"
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

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Get the number of available GPUs
    available_gpus = get_available_gpu_count()

    profile_tp_size = [2**i for i in range(int(math.log2(available_gpus)) + 1)]
    logger.info(f"Profiling TP sizes: {profile_tp_size}")

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = get_model_name(config)
    port = get_port(config)

    # first profile prefill
    prefill_tp_size = []
    prefill_ttft = []
    prefill_thpt_per_gpu = []
    logger.info("Profiling prefill...")
    prefill_config = convert_config(config, "prefill")
    for tp_size in profile_tp_size:
        logger.info(f"Profiling prefill with TP size {tp_size}...")
        prefill_config = set_config_tp_size(prefill_config, tp_size)
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
        plt.figure(figsize=(10, 6))
        plt.scatter(prefill_ttft, prefill_thpt_per_gpu, s=100)
        for i, tp in enumerate(prefill_tp_size):
            plt.annotate(
                f"TP{tp}",
                (prefill_ttft[i], prefill_thpt_per_gpu[i]),
                xytext=(10, 0),
                textcoords="offset points",
                fontsize=10,
            )

        plt.axvline(
            x=args.ttft, color="r", linestyle="--", label=f"Target TTFT: {args.ttft} ms"
        )
        plt.legend()

        plt.title("Prefill Performance")
        plt.xlabel("Time to First Token (ms)")
        plt.ylabel("Prefill throughput per GPU (tokens/s/GPU)")
        plt.grid(True)

        plot_path = f"{args.output_dir}/prefill_performance.png"
        plt.savefig(plot_path, dpi=300)
        logger.info(f"Performance plot saved to {plot_path}")
        plt.close()

    # then profile decode
    plt.figure(figsize=(10, 6))
    decode_tp_size = []
    decode_itl = []
    decode_thpt_per_gpu = []
    decode_concurrency = []
    decode_kv_cache_size = []
    logger.info("Profiling decode...")
    decode_config = convert_config(config, "decode")
    for tp_size in profile_tp_size:
        logger.info(f"Profiling decode with TP size {tp_size}...")
        decode_config = set_config_tp_size(decode_config, tp_size)
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
                preexec_fn=os.setsid,  # Use process group for clean termination
            )

        if not wait_for_server_ready(model_name, port):
            logger.error(f"Server did not become ready, skip profiling tp={tp_size}")
            break

        max_kv_tokens = get_kv_cache_size_from_dynamo_log(dynamo_log_fn)
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

        # Plot a line in the 2d plot
        plt.plot(engine_decode_itl, engine_decode_thpt_per_gpu, label=f"TP{tp_size}")

    plt.axvline(
        x=args.itl, color="r", linestyle="--", label=f"Target ITL: {args.itl} ms"
    )
    plt.legend()
    plt.title("Decode Performance")
    plt.xlabel("Inter Token Latency (ms)")
    plt.ylabel("Decode throughput per GPU (tokens/s/GPU)")
    plt.grid(True)

    plot_path = f"{args.output_dir}/decode_performance.png"
    plt.savefig(plot_path, dpi=300)
    logger.info(f"Performance plot saved to {plot_path}")
    plt.close()

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
    prefill_config = convert_config(config, "prefill")
    prefill_config = set_config_tp_size(prefill_config, tp_size)
    logger.info(f"Dynamo config: {prefill_config}")

    work_dir = f"{args.output_dir}/prefill_tp{tp_size}_interpolation"
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

        # Fit quadratic functions
        ttft_coeffs = np.polyfit(prefill_isl_np, prefill_ttft_np, 2)
        thpt_coeffs = np.polyfit(prefill_isl_np, prefill_thpt_per_gpu_np, 2)

        # Create interpolation functions
        ttft_poly = np.poly1d(ttft_coeffs)
        thpt_poly = np.poly1d(thpt_coeffs)

        # Generate points for smooth curves
        x_interp = np.linspace(min(prefill_isl_np), max(prefill_isl_np), 100)
        ttft_interp = ttft_poly(x_interp)
        thpt_interp = thpt_poly(x_interp)

        # Plot TTFT vs ISL
        plt.figure(figsize=(10, 6))
        plt.scatter(prefill_isl_np, prefill_ttft_np, s=100, label="Measured data")
        plt.plot(
            x_interp,
            ttft_interp,
            "r-",
            label=f"Quadratic fit: {ttft_coeffs[0]:.2e}x² + {ttft_coeffs[1]:.2e}x + {ttft_coeffs[2]:.2e}",
        )

        plt.title("Prefill TTFT vs Input Sequence Length")
        plt.xlabel("Input Sequence Length (tokens)")
        plt.ylabel("Time to First Token (ms)")
        plt.grid(True)
        plt.legend()

        ttft_plot_path = f"{work_dir}/prefill_ttft_interpolation.png"
        plt.savefig(ttft_plot_path, dpi=300)
        logger.info(f"TTFT interpolation plot saved to {ttft_plot_path}")
        plt.close()

        # Plot Throughput vs ISL
        plt.figure(figsize=(10, 6))
        plt.scatter(
            prefill_isl_np, prefill_thpt_per_gpu_np, s=100, label="Measured data"
        )
        plt.plot(
            x_interp,
            thpt_interp,
            "g-",
            label=f"Quadratic fit: {thpt_coeffs[0]:.2e}x² + {thpt_coeffs[1]:.2e}x + {thpt_coeffs[2]:.2e}",
        )

        plt.title("Prefill Throughput vs Input Sequence Length")
        plt.xlabel("Input Sequence Length (tokens)")
        plt.ylabel("Prefill throughput per GPU (tokens/s/GPU)")
        plt.grid(True)
        plt.legend()

        thpt_plot_path = f"{work_dir}/prefill_throughput_interpolation.png"
        plt.savefig(thpt_plot_path, dpi=300)
        logger.info(
            f"Prefill throughput per GPU interpolation plot saved to {thpt_plot_path}"
        )
        plt.close()
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
    decode_config = set_config_tp_size(decode_config, best_decode_tp)
    logger.info(f"Dynamo config: {decode_config}")

    work_dir = f"{args.output_dir}/decode_tp{best_decode_tp}_interpolation"
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
            preexec_fn=os.setsid,  # Use process group for clean termination
        )

    if not wait_for_server_ready(model_name, port):
        logger.error(f"Server did not become ready, skip profiling tp={tp_size}")
    else:
        max_kv_tokens = get_kv_cache_size_from_dynamo_log(dynamo_log_fn)

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
        save_path = f"{work_dir}/decode_tp{tp_size}_data.npz"
        np.savez(
            save_path,
            x_kv_usage=np.array(x_kv_usage),
            y_context_length=np.array(y_context_length),
            z_itl=np.array(z_itl),
            z_thpt_per_gpu=np.array(z_thpt_per_gpu),
        )
        logger.info(f"Saved data points to {save_path}")

        xi = np.linspace(min(x_kv_usage), max(x_kv_usage), 100)
        yi = np.linspace(min(y_context_length), max(y_context_length), 100)
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((x_kv_usage, y_context_length), z_itl, (X, Y), method="cubic")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")  # type: ignore

        # Create the surface plot with customizations
        surf = ax.plot_surface(  # type: ignore
            X,
            Y,
            Z,
            cmap=cm.coolwarm,  # type: ignore
            linewidth=0.2,
            antialiased=True,
            alpha=0.8,
        )

        # Add a color bar with custom settings
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Z Value", fontsize=12)
        cbar.ax.tick_params(labelsize=10)

        # Add labels with custom font sizes
        ax.set_xlabel("Active KV Percentage", fontsize=12)
        ax.set_ylabel("Decode Context Length", fontsize=12)
        ax.set_zlabel("ITL", fontsize=12)  # type: ignore

        # Set viewing angle
        ax.view_init(elev=30, azim=45)  # type: ignore
        ax.grid(True)
        ax.tick_params(axis="both", which="major", labelsize=10)

        logger.info(f"Saving ITL surface plot to {work_dir}/decode_tp{tp_size}.png")
        plt.savefig(f"{work_dir}/decode_tp{tp_size}.png", dpi=300, bbox_inches="tight")
