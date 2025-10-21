# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test gRPC parameter passing with tensor models."""

import logging
import os
import shutil

import numpy as np
import pytest
import tritonclient.grpc as grpcclient

from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


class DynamoFrontendProcess(ManagedProcess):
    def __init__(self, request):
        command = ["python", "-m", "dynamo.frontend", "--kserve-grpc-server"]
        log_dir = f"{request.node.name}_frontend"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=command,
            display_output=True,
            terminate_existing=True,
            log_dir=log_dir,
        )


class TensorParamsWorkerProcess(ManagedProcess):
    def __init__(self, request):
        command = [
            "python3",
            os.path.join(os.path.dirname(__file__), "tensor_params_worker.py"),
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_ENABLED"] = "true"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = "8083"

        log_dir = f"{request.node.name}_worker"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (
                    "http://localhost:8083/health",
                    lambda r: r.json().get("status") == "ready",
                )
            ],
            timeout=300,
            display_output=True,
            log_dir=log_dir,
        )


@pytest.fixture()
def start_services(request, runtime_services):
    """Start frontend and worker with fresh etcd/nats."""
    with DynamoFrontendProcess(request):
        with TensorParamsWorkerProcess(request):
            yield


def extract_params(param_map) -> dict:
    """Extract parameters from gRPC response."""
    result = {}
    for key, param in param_map.items():
        for field in [
            "bool_param",
            "int64_param",
            "double_param",
            "string_param",
            "uint64_param",
        ]:
            if param.HasField(field):
                result[key] = getattr(param, field)
                break
    return result


@pytest.mark.e2e
@pytest.mark.pre_merge
@pytest.mark.parametrize(
    "request_params",
    [
        None,
        {"int_param": 8},
        {"str_param": "custom", "bool_param": True},
    ],
    ids=["no_params", "numeric_param", "mixed_params"],
)
def test_request_parameters(start_services, request_params):
    """Test gRPC request-level parameters are echoed through tensor models.

    The worker acts as an identity function: echoes input tensors unchanged and
    returns all request parameters plus a "processed" flag to verify the complete
    parameter flow through the gRPC frontend.
    """
    client = grpcclient.InferenceServerClient("localhost:8000")

    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    inputs = [grpcclient.InferInput("INPUT", input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)

    response = client.infer(
        "tensor-params-model", inputs=inputs, parameters=request_params
    )

    response_msg = response.get_response()
    output_tensor = response_msg.outputs[0]
    output_data = np.array(list(output_tensor.contents.fp32_contents), dtype=np.float32)
    assert np.array_equal(input_data, output_data)

    resp_params = extract_params(response_msg.parameters)

    assert resp_params.get("processed") is True

    if request_params:
        for key, expected_value in request_params.items():
            assert key in resp_params, f"Parameter '{key}' not echoed"
            actual = resp_params[key]
            assert (
                actual == expected_value
            ), f"{key}: expected {expected_value}, got {actual}"
