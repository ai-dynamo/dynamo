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

import pytest

from dynamo.sdk.lib.config import ServiceConfig

pytestmark = pytest.mark.pre_merge


def test_service_config_with_common_configs():
    # Reset singleton instance
    ServiceConfig._instance = None

    # Set environment variable with config that includes common-configs
    os.environ[
        "DYNAMO_SERVICE_CONFIG"
    ] = """
    {
        "Common": {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "block-size": 64,
            "max-model-len": 16384
        },
        "VllmWorker": {
            "enforce-eager": true,
            "common-configs": ["model", "block-size", "max-model-len"]
        }
    }
    """

    # Get arguments and verify common configs are included
    service_config = ServiceConfig.get_instance()
    vllm_worker_args = service_config.as_args("VllmWorker")

    # Check that each common config appears in the arguments
    for key in ["model", "block-size", "max-model-len"]:
        assert f"--{key}" in vllm_worker_args


def test_service_config_without_common_configs():
    # Reset singleton instance
    ServiceConfig._instance = None

    # Set environment variable with config that DOESN'T include common-configs
    os.environ[
        "DYNAMO_SERVICE_CONFIG"
    ] = """
    {
        "Common": {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "block-size": 64,
            "max-model-len": 16384
        },
        "VllmWorker": {
            "enforce-eager": true
        }
    }
    """

    # Get arguments and verify common configs are NOT included
    service_config = ServiceConfig.get_instance()
    vllm_worker_args = service_config.as_args("VllmWorker")

    # Check that none of the common configs appear in arguments
    for key in ["model", "block-size", "max-model-len"]:
        assert f"--{key}" not in vllm_worker_args


def test_service_config_with_direct_configs():
    # Reset singleton instance
    ServiceConfig._instance = None

    # Set environment variable with direct configs (no Common section reference)
    os.environ[
        "DYNAMO_SERVICE_CONFIG"
    ] = """
    {
        "VllmWorker": {
            "enforce-eager": true,
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "block-size": 64,
            "max-model-len": 16384
        }
    }
    """

    # Get arguments and verify direct configs are included
    service_config = ServiceConfig.get_instance()
    vllm_worker_args = service_config.as_args("VllmWorker")

    # Check that each config appears in the arguments
    for key in ["model", "block-size", "max-model-len"]:
        assert f"--{key}" in vllm_worker_args


def test_service_config_override_common_configs():
    # Reset singleton instance
    ServiceConfig._instance = None

    # Set environment variable with config that includes common-configs
    # overridden by the subscribing config
    os.environ[
        "DYNAMO_SERVICE_CONFIG"
    ] = """
    {
        "Common": {
            "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "block-size": 64,
            "max-model-len": 16384
        },
        "VllmWorker": {
            "enforce-eager": true,
            "block-size": 128,
            "common-configs": ["model", "block-size", "max-model-len"]
        }
    }
    """

    # Get arguments and verify common configs are included
    service_config = ServiceConfig.get_instance()
    vllm_worker_args = service_config.as_args("VllmWorker")

    # Check that each common config appears in the arguments
    for key in ["model", "block-size", "max-model-len"]:
        assert f"--{key}" in vllm_worker_args

    assert vllm_worker_args[vllm_worker_args.index("--block-size") + 1] == "128"


def test_explicit_boolean_arguments():
    """Test that boolean arguments are handled correctly with new logic"""
    # Reset singleton instance
    ServiceConfig._instance = None

    # Set environment variable with boolean configs
    os.environ[
        "DYNAMO_SERVICE_CONFIG"
    ] = """
    {
        "VllmWorker": {
            "enable-prefix-caching": true,
            "disable-sliding-window": false,
            "enforce-eager": true
        }
    }
    """

    # Get arguments and verify boolean handling
    service_config = ServiceConfig.get_instance()
    vllm_worker_args = service_config.as_args("VllmWorker")

    # Check that true values are passed as flags only
    assert "--enable-prefix-caching" in vllm_worker_args
    # Should NOT have a following "true" value
    enable_idx = vllm_worker_args.index("--enable-prefix-caching")
    assert (
        enable_idx == len(vllm_worker_args) - 1
        or not vllm_worker_args[enable_idx + 1] == "true"
    )

    # Check that false values for standard boolean flags are omitted
    assert "--disable-sliding-window" not in vllm_worker_args

    # Check that another true value works as flag
    assert "--enforce-eager" in vllm_worker_args
    enforce_idx = vllm_worker_args.index("--enforce-eager")
    assert (
        enforce_idx == len(vllm_worker_args) - 1
        or not vllm_worker_args[enforce_idx + 1] == "true"
    )


def test_vllm_boolean_arguments_special_handling():
    """Test that vLLM boolean arguments with special defaults are handled correctly"""
    # Reset singleton instance
    ServiceConfig._instance = None

    # Set environment variable with vLLM boolean configs
    os.environ[
        "DYNAMO_SERVICE_CONFIG"
    ] = """
    {
        "VllmWorker": {
            "enable-prefix-caching": false,
            "use-tqdm-on-load": false,
            "multi-step-stream-outputs": false,
            "some-other-flag": false
        }
    }
    """

    # Get arguments and verify vLLM special boolean handling
    service_config = ServiceConfig.get_instance()
    vllm_worker_args = service_config.as_args("VllmWorker")

    # Check that enable-prefix-caching false uses negative flag
    assert "--no-enable-prefix-caching" in vllm_worker_args
    assert "--enable-prefix-caching" not in vllm_worker_args

    # Check that use-tqdm-on-load false uses negative flag
    assert "--no-use-tqdm-on-load" in vllm_worker_args
    assert "--use-tqdm-on-load" not in vllm_worker_args

    # Check that multi-step-stream-outputs false uses negative flag
    assert "--no-multi-step-stream-outputs" in vllm_worker_args
    assert "--multi-step-stream-outputs" not in vllm_worker_args

    # Check that other false flags are omitted (standard behavior)
    assert "--some-other-flag" not in vllm_worker_args
