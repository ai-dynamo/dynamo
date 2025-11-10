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
        "3",
        "--artifact-dir",
        artifact_dir,
        "--random-seed",
        str(seed),
    ]


def get_prefill_aiperf_cmd(
    isl,
    artifact_dir,
    seed=100,
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    tokenizer="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    osl=5,
    base_url="http://localhost:8000",
):
    return _get_common_aiperf_cmd(
        artifact_dir,
        seed,
        model,
        tokenizer,
        base_url,
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
        "1",
        "--request-count",
        "1",
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
):
    return _get_common_aiperf_cmd(
        artifact_dir,
        seed,
        model,
        tokenizer,
        base_url,
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
):
    logger.info(f"Running aiperf with isl {isl}")
    aiperf_cmd = get_prefill_aiperf_cmd(
        isl,
        aiperf_artifact_dir,
        model=model_name,
        tokenizer=tokenizer,
        base_url=base_url,
    )
    print(f"aiperf cmd: {aiperf_cmd}")
    # import pdb; pdb.set_trace()
    aiperf_process = subprocess.Popen(
        aiperf_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    stdout, stderr = aiperf_process.communicate()
    if aiperf_process.returncode == 0:
        logger.info("AIperf profiling completed successfully")
        logger.info(stdout)
        aiperf_result = get_aiperf_result(aiperf_artifact_dir)
        return aiperf_result
    else:
        logger.error(f"AIPerf failed with error code: {aiperf_process.returncode}")
        logger.error(f"stderr: {stderr}")
        return None


def benchmark_decode(
    isl,
    osl,
    num_request,
    aiperf_artifact_dir,
    model_name,
    tokenizer,
    base_url="http://localhost:8000",
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
        logger.info(stdout)
        aiperf_result = get_aiperf_result(aiperf_artifact_dir)
        return aiperf_result
    else:
        logger.error(f"AIPerf failed with error code: {aiperf_process.returncode}")
        logger.error(f"stderr: {stderr}")
        return None
