# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parallelization: Hermetic test (xdist-safe via dynamic ports).
# Tested on: Linux (Ubuntu 24.04 container), Intel(R) Core(TM) i9-14900K, 32 vCPU.
# Combined pre_merge wall time (this file + test_tensor_mocker_engine.py):
# - Serialized: 87.48s.
# - Parallel (-n auto): 25.27s (62.21s saved, 3.46x).
# GPU Requirement: gpu_0 (CPU-only, tensor echo worker does not use GPU)

"""Test gRPC parameter passing with tensor models."""

import logging
import os
from pathlib import Path

import numpy as np
import pytest
from packaging.requirements import Requirement

try:
    from google.protobuf import any_pb2, empty_pb2, json_format
except ImportError:
    any_pb2 = empty_pb2 = json_format = None

try:
    import tritonclient.grpc as grpcclient
    from tritonclient.grpc import service_pb2
except ImportError:
    grpcclient = service_pb2 = None

from tests.utils.managed_process import ManagedProcess, check_health_ready

logger = logging.getLogger(__name__)


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.parallel
@pytest.mark.parametrize("component", ["common", "frontend", "planner"])
def test_protobuf_requirements_exclude_vulnerable_versions(component: str) -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "container/deps"
        / f"requirements.{component}.txt"
    )
    requirement = Requirement(
        next(
            line
            for line in path.read_text().splitlines()
            if line.startswith("protobuf")
        )
    )
    assert "6.33.6" in requirement.specifier
    for version in ("5.29.5", "6.30.0", "6.31.1", "6.32.1", "6.33.4", "7.0.0"):
        assert version not in requirement.specifier, f"{path} allows protobuf {version}"


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.parallel
@pytest.mark.skipif(json_format is None, reason="protobuf is not installed")
def test_protobuf_any_json_recursion_limit() -> None:
    message = any_pb2.Any()
    message.Pack(empty_pb2.Empty())
    payload = json_format.MessageToDict(message)
    assert (
        json_format.ParseDict(payload, any_pb2.Any(), max_recursion_depth=5) == message
    )
    for _ in range(10):
        payload = {"@type": "type.googleapis.com/google.protobuf.Any", "value": payload}
    with pytest.raises(json_format.ParseError, match="[Rr]ecursion"):
        json_format.ParseDict(payload, any_pb2.Any(), max_recursion_depth=5)


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.parallel
@pytest.mark.skipif(grpcclient is None, reason="tritonclient.grpc is not installed")
def test_triton_protobuf_json_roundtrip() -> None:
    response = service_pb2.ModelInferResponse(model_name="identity", id="roundtrip")
    response.parameters["processed"].bool_param = True
    response.outputs.add(name="OUTPUT", datatype="INT32", shape=[2])
    response.raw_output_contents.append(np.array([3, 7], dtype=np.int32).tobytes())
    wire_response = service_pb2.ModelInferResponse.FromString(
        response.SerializeToString()
    )
    result = grpcclient.InferResult(wire_response)
    assert result.get_response(as_json=True)["parameters"]["processed"]["bool_param"]
    assert result.get_response().id == "roundtrip"
    np.testing.assert_array_equal(result.as_numpy("OUTPUT"), [3, 7])


class EchoTensorWorkerProcess(ManagedProcess):
    def __init__(self, request, system_port: int):
        self.system_port = system_port

        command = [
            "python3",
            os.path.join(os.path.dirname(__file__), "echo_tensor_worker.py"),
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(system_port)
        # Each test gets its own Etcd/NATS from runtime_services_dynamic_ports,
        # so no namespace conflicts - use default "tensor" namespace

        log_dir = f"{request.node.name}_worker"
        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (f"http://localhost:{system_port}/health", check_health_ready)
            ],
            timeout=300,
            display_output=True,
            log_dir=log_dir,
            terminate_all_matching_process_names=False,
        )


@pytest.fixture(scope="function")
def start_services_with_echo_tensor_worker(request, start_services_with_grpc):
    """Start echo tensor worker with the shared gRPC frontend.

    Function-scoped to allow parallel test execution.
    Each test gets its own gRPC frontend + echo tensor worker on unique ports.
    No namespace conflicts because runtime_services_dynamic_ports provides isolated Etcd/NATS.
    """
    frontend_port, system_port = start_services_with_grpc
    with EchoTensorWorkerProcess(request, system_port):
        logger.info(f"Echo Tensor Worker started for test on port {frontend_port}")
        yield frontend_port


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
@pytest.mark.gpu_0  # Echo tensor worker is CPU-only (no GPU required)
@pytest.mark.parallel
@pytest.mark.parametrize(
    "request_params",
    [
        None,
        {"int_param": 8},
        {"str_param": "custom", "bool_param": True},
    ],
    ids=["no_params", "numeric_param", "mixed_params"],
)
def test_request_parameters(
    file_storage_backend, start_services_with_echo_tensor_worker, request_params
):
    """Test gRPC request-level parameters are echoed through tensor models.

    The worker acts as an identity function: echoes input tensors unchanged and
    returns all request parameters plus a "processed" flag to verify the complete
    parameter flow through the gRPC frontend.
    """
    frontend_port = start_services_with_echo_tensor_worker
    client = grpcclient.InferenceServerClient(f"localhost:{frontend_port}")

    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    inputs = [grpcclient.InferInput("INPUT", input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)

    response = client.infer("echo", inputs=inputs, parameters=request_params)

    output_data = response.as_numpy("INPUT")
    assert output_data is not None, "Expected response to include output tensor 'INPUT'"
    assert np.array_equal(input_data, output_data)

    response_msg = response.get_response()

    resp_params = extract_params(response_msg.parameters)

    assert resp_params.get("processed") is True

    if request_params:
        for key, expected_value in request_params.items():
            assert key in resp_params, f"Parameter '{key}' not echoed"
            actual = resp_params[key]
            assert (
                actual == expected_value
            ), f"{key}: expected {expected_value}, got {actual}"
