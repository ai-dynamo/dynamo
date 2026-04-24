# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
import os

import pytest

from tests.serve.common import (
    WORKSPACE_DIR,
    params_with_model_mark,
    run_serve_deployment,
)
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig
from tests.utils.payload_builder import chat_payload_default, completion_payload_default

logger = logging.getLogger(__name__)

sample_dir = os.path.join(WORKSPACE_DIR, "examples/backends/sample")

sample_configs = {
    "aggregated": EngineConfig(
        name="aggregated",
        directory=sample_dir,
        script_name="agg.sh",
        script_args=["--model-name", "Qwen/Qwen3-0.6B"],
        marks=[
            pytest.mark.gpu_0,
            pytest.mark.timeout(300),
            pytest.mark.pre_merge,
        ],
        model="Qwen/Qwen3-0.6B",
        frontend_port=DefaultPort.FRONTEND.value,
        request_payloads=[
            chat_payload_default(),
            completion_payload_default(),
        ],
    ),
}


@pytest.fixture(params=params_with_model_mark(sample_configs))
def sample_config_test(request):
    """Fixture that provides different sample test configurations"""
    return sample_configs[request.param]


@pytest.mark.e2e
def test_sample_deployment(
    sample_config_test,
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_models,
):
    """Test sample backend deployment using the unified Worker."""
    config = dataclasses.replace(
        sample_config_test, frontend_port=dynamo_dynamic_ports.frontend_port
    )
    run_serve_deployment(config, request, ports=dynamo_dynamic_ports)
