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

import json
import logging
import os
import random
import subprocess
from typing import Optional, Tuple

from benchmarks.profiler.utils.defaults import (
    AIPERF_PREFILL_ATTN_DP_NUM_REQ_RATIO,
    AIPERF_PREFILL_BENCHMARK_OSL,
    AIPERF_WARMUP_REQUEST_PER_DP_RANK,
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


def _get_common_aiperf_cmd(
    artifact_dir,
    seed=100,
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    tokenizer="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    base_url="http://localhost:8000",
    warmup_request_count: int = AIPERF_WARMUP_REQUEST_PER_DP_RANK,
):
    return [
        "aiperf",
        "profile",
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--endpoint-type",
        "chat",
        "--endpoint",
        "/v1/chat/completions",
        "--streaming",
        "--url",
        base_url,
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        '{"nvext":{"ignore_eos":true}}',
        "--warmup-request-count",
        str(warmup_request_count),
        "--artifact-dir",
        artifact_dir,
        "--random-seed",
        str(seed),
        "--request-timeout-seconds",
        "1800",
    ]


def get_prefill_aiperf_cmd(
    isl,
    artifact_dir,
    seed=100,
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    tokenizer="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    osl=AIPERF_PREFILL_BENCHMARK_OSL,
    base_url="http://localhost:8000",
    concurrency: int = 1,
    request_count: int = 1,
    warmup_request_count: int = AIPERF_WARMUP_REQUEST_PER_DP_RANK,
):
    return _get_common_aiperf_cmd(
        artifact_dir,
        seed,
        model,
        tokenizer,
        base_url,
        warmup_request_count=warmup_request_count,
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
        str(concurrency),
        "--request-count",
        str(request_count),
    ]


def get_decode_aiperf_cmd(
    isl,
    osl,
    artifact_dir,
    num_request,
    seed=100,
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    tokenizer="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    base_url="http://localhost:8000",
    warmup_request_count: int = AIPERF_WARMUP_REQUEST_PER_DP_RANK,
):
    return _get_common_aiperf_cmd(
        artifact_dir,
        seed,
        model,
        tokenizer,
        base_url,
        warmup_request_count=warmup_request_count,
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


def get_aiperf_result(artifact_dir: str) -> dict:
    json_file_path = None
    for root, _, files in os.walk(artifact_dir):
        if "profile_export_aiperf.json" in files:
            json_file_path = os.path.join(root, "profile_export_aiperf.json")
            break
    if json_file_path is None:
        raise FileNotFoundError(
            f"profile_export_aiperf.json not found in {artifact_dir}"
        )
    with open(json_file_path, "r") as f:
        return json.load(f)


def benchmark_prefill(
    isl,
    aiperf_artifact_dir,
    model_name,
    tokenizer,
    base_url="http://localhost:8000",
    concurrency: int = 1,
    request_count: int = 1,
    warmup_request_count: int = 3,
):
    logger.info(f"Running aiperf with isl {isl}")
    aiperf_cmd = get_prefill_aiperf_cmd(
        isl,
        aiperf_artifact_dir,
        model=model_name,
        tokenizer=tokenizer,
        base_url=base_url,
        concurrency=concurrency,
        request_count=request_count,
        warmup_request_count=warmup_request_count,
    )
    logger.debug(f"aiperf cmd: {aiperf_cmd}")

    aiperf_process = subprocess.Popen(
        aiperf_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = aiperf_process.communicate()
    if aiperf_process.returncode == 0:
        logger.info("AIperf profiling completed successfully")
        logger.debug(stdout)
        aiperf_result = get_aiperf_result(aiperf_artifact_dir)
        return aiperf_result
    else:
        logger.error(f"AIPerf failed with error code: {aiperf_process.returncode}")
        logger.error(f"stderr: {stderr}")
        return None


def get_prefill_ttft(
    isl: int,
    aiperf_artifact_dir: str,
    model_name: str,
    tokenizer: str,
    base_url: str = "http://localhost:8000",
    attention_dp_size: int = 1,
    attn_dp_num_req_ratio: int = AIPERF_PREFILL_ATTN_DP_NUM_REQ_RATIO,
) -> Optional[float]:
    """
    Run prefill benchmark and extract TTFT (ms). Returns None on failure.
    If attention_dp_size > 1 (DEP), send attn_dp_size * attn_dp_num_req_ratio concurrent requests (single burst),
    then compute TTFT as (max TTFT across burst) / attn_dp_num_req_ratio.
    attn_dp_num_req_ratio defaults to 4 rounds to account for the error margin caused
    by the first batch being launched too early without enough requests.
    """
    # DEP-aware measurement (waves of size attention_dp_size)
    if attention_dp_size > 1:
        assert attn_dp_num_req_ratio > 0, "attn_dp_num_req_ratio must be greater than 0"
        total_concurrency = attention_dp_size * attn_dp_num_req_ratio
        logger.info(
            f"DEP prefill measurement: isl={isl}, attn_dp={attention_dp_size}, attn_dp_num_req_ratio={attn_dp_num_req_ratio}, "
            f"total_concurrency={total_concurrency}"
        )
        # Run aiperf with the requested concurrency; allow normal warmup behavior
        aiperf_result = benchmark_prefill(
            isl,
            aiperf_artifact_dir,
            model_name,
            tokenizer,
            base_url=base_url,
            concurrency=total_concurrency,
            request_count=total_concurrency,
            warmup_request_count=AIPERF_WARMUP_REQUEST_PER_DP_RANK * attention_dp_size,
        )
        try:
            max_ttft = float(aiperf_result["time_to_first_token"]["max"])
            # subtract the decoding time in-between prefill runs
            max_ttft -= (
                float(aiperf_result["inter_token_latency"]["avg"])
                * (AIPERF_PREFILL_BENCHMARK_OSL - 1)
                * (attn_dp_num_req_ratio - 1)
            )
            return max_ttft / float(attn_dp_num_req_ratio)
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "Failed to extract max TTFT from AIPerf result for DEP prefill"
            )
            return None

    # Default path (non-DEP): use AIPerf's TTFT metric
    aiperf_result = benchmark_prefill(
        isl,
        aiperf_artifact_dir,
        model_name,
        tokenizer,
        base_url=base_url,
    )
    try:
        return float(aiperf_result["time_to_first_token"]["avg"])
    except (KeyError, TypeError, ValueError):
        logger.warning("Failed to extract TTFT from AIPerf result")
        return None


def get_decode_itl_and_thpt_per_gpu(
    isl: int,
    osl: int,
    num_request: int,
    aiperf_artifact_dir: str,
    model_name: str,
    tokenizer: str,
    base_url: str = "http://localhost:8000",
    num_gpus: int = 1,
    attention_dp_size: int = 1,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Run decode benchmark and extract (ITL ms, throughput per GPU).
    Returns (None, None) on failure.
    """
    aiperf_result = benchmark_decode(
        isl,
        osl,
        num_request,
        aiperf_artifact_dir,
        model_name,
        tokenizer,
        base_url=base_url,
        warmup_request_count=AIPERF_WARMUP_REQUEST_PER_DP_RANK * attention_dp_size,
    )
    if aiperf_result is None:
        return None, None
    try:
        itl = float(aiperf_result["inter_token_latency"]["avg"])
        thpt_total = float(aiperf_result["output_token_throughput"]["avg"])
        thpt_per_gpu = thpt_total / max(num_gpus, 1)
        return itl, thpt_per_gpu
    except (KeyError, TypeError, ValueError):
        logger.warning("Failed to extract decode metrics from AIPerf result")
        return None, None


def benchmark_decode(
    isl,
    osl,
    num_request,
    aiperf_artifact_dir,
    model_name,
    tokenizer,
    base_url="http://localhost:8000",
    warmup_request_count: int = AIPERF_WARMUP_REQUEST_PER_DP_RANK,
):
    logger.info(f"Profiling decode with num_request {num_request}...")

    # first warm-up the engine by pre-computing all prefill tokens
    # we use the same random seed to make sure the prompt is the same
    seed = random.randint(0, 1000000)

    aiperf_cmd = get_decode_aiperf_cmd(
        isl,
        osl,
        f"{aiperf_artifact_dir}_warmup",
        num_request,
        seed=seed,
        model=model_name,
        tokenizer=tokenizer,
        base_url=base_url,
        warmup_request_count=warmup_request_count,
    )
    aiperf_process = subprocess.Popen(
        aiperf_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    aiperf_process.communicate()
    # then send out the real requests, hopefully, this will skip all prefill computation
    aiperf_cmd = get_decode_aiperf_cmd(
        isl,
        osl,
        aiperf_artifact_dir,
        num_request,
        seed=seed,
        model=model_name,
        tokenizer=tokenizer,
        base_url=base_url,
    )
    aiperf_process = subprocess.Popen(
        aiperf_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = aiperf_process.communicate()
    if aiperf_process.returncode == 0:
        logger.info("AIperf profiling completed successfully")
        logger.debug(stdout)
        aiperf_result = get_aiperf_result(aiperf_artifact_dir)
        return aiperf_result
    else:
        logger.error(f"AIPerf failed with error code: {aiperf_process.returncode}")
        logger.error(f"stderr: {stderr}")
        return None
