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

import os

from dynamo.sdk.lib.config import ServiceConfig


def test_service_config_subscribed_correctly():
    ServiceConfig._instance = None
    os.environ[
        "DYNAMO_SERVICE_CONFIG"
    ] = """
{"Common": {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "block-size": 64, "max-model-len": 16384}, "Frontend": {"served_model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "endpoint": "dynamo.Processor.chat/completions", "port": 8000}, "Processor": {"router": "round-robin", "common-configs": ["model", "block-size", "max-model-len"]}, "VllmWorker": {"enforce-eager": true, "max-num-batched-tokens": 16384, "enable-prefix-caching":
true, "router": "random", "tensor-parallel-size": 1, "ServiceArgs": {"workers": 1}, "common-configs": ["model", "block-size", "max-model-len"]}}
"""
    service_config = ServiceConfig.get_instance()
    vllm_worker_args = service_config.as_args("VllmWorker")
    assert all(
        any(arg == f"--{key}" for arg in vllm_worker_args)
        for key in ["model", "block-size", "max-model-len"]
    )


def test_service_config_not_subscribed():
    ServiceConfig._instance = None
    os.environ[
        "DYNAMO_SERVICE_CONFIG"
    ] = """
{"Common": {"model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "block-size": 64, "max-model-len": 16384}, "Frontend": {"served_model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "endpoint": "dynamo.Processor.chat/completions", "port": 8000}, "Processor": {"router": "round-robin", "common-configs": ["model", "block-size", "max-model-len"]}, "VllmWorker": {"enforce-eager": true, "max-num-batched-tokens": 16384, "enable-prefix-caching":
true, "router": "random", "tensor-parallel-size": 1, "ServiceArgs": {"workers": 1}}}
"""
    service_config = ServiceConfig.get_instance()
    vllm_worker_args = service_config.as_args("VllmWorker")
    assert all(
        not arg == f"--{key}"
        for key in ["model", "block-size", "max-model-len"]
        for arg in vllm_worker_args
    )


def test_service_config_no_common_config():
    ServiceConfig._instance = None
    os.environ[
        "DYNAMO_SERVICE_CONFIG"
    ] = """
{"Frontend": {"served_model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "endpoint": "dynamo.Processor.chat/completions", "port": 8000}, "Processor": {"router": "round-robin", "common-configs": ["model", "block-size", "max-model-len"]}, "VllmWorker": {"enforce-eager": true, "max-num-batched-tokens": 16384, "enable-prefix-caching":
true, "router": "random", "tensor-parallel-size": 1, "ServiceArgs": {"workers": 1}, "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "block-size": 64, "max-model-len": 16384]}}
"""
    service_config = ServiceConfig.get_instance()
    vllm_worker_args = service_config.as_args("VllmWorker")
    assert all(
        any(arg == f"--{key}" for arg in vllm_worker_args)
        for key in ["model", "block-size", "max-model-len"]
    )
