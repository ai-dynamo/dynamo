# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parallelization: Hermetic test (xdist-safe via dynamic ports).
# Tested on: Linux (Ubuntu 24.04 container), Intel(R) Core(TM) i9-14900K, 32 vCPU.
# Combined pre_merge wall time (this file + test_tensor_mocker_engine.py):
# - Serialized: 87.48s.
# - Parallel (-n auto): 25.27s (62.21s saved, 3.46x).
# GPU Requirement: gpu_0 (CPU-only, tensor echo worker does not use GPU)

"""Test gRPC parameter passing with tensor models."""

import builtins
import logging
import os
import runpy
import shlex
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from tests.utils.managed_process import ManagedProcess, check_health_ready

try:
    from google.protobuf import any_pb2, empty_pb2, json_format
except ImportError:
    any_pb2 = empty_pb2 = json_format = None

TRITON_SKIP_REASON = "tritonclient.grpc is not installed"
try:
    import tritonclient.grpc as grpcclient
    from tritonclient.grpc import service_pb2
except ImportError:
    grpcclient = service_pb2 = None
except RuntimeError as exc:
    if not (
        str(exc).startswith("The grpc package installed is at version ")
        and "grpc_service_pb2_grpc.py depends on grpcio>=" in str(exc)
    ):
        raise
    grpcclient = service_pb2 = None
    TRITON_SKIP_REASON = str(exc)

logger = logging.getLogger(__name__)


def _requirement(text: str, package: str) -> Requirement:
    pins = []
    for line in text.splitlines():
        value = line.partition("#")[0].strip()
        if not value or value.startswith("-"):
            continue
        requirement = Requirement(value)
        if requirement.name == package:
            pins.append(requirement)
    assert len(pins) == 1, f"Expected one {package} requirement, found {len(pins)}"
    return pins[0]


def _assert_triton_pins_match(frontend: str, dockerfile: str) -> None:
    packages = ("tritonclient", "protobuf", "grpcio")
    tokens = [
        token
        for line in dockerfile.replace("\\\n", " ").splitlines()
        if line.startswith("RUN uv pip install ")
        for token in shlex.split(line)[4:]
        if token.startswith(packages)
    ]
    for package in packages:
        actual = _requirement("\n".join(tokens), package)
        expected = _requirement(frontend, package)
        assert actual == expected, f"Triton example {actual} differs from {expected}"


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
    requirement = _requirement(path.read_text(encoding="utf-8"), "protobuf")
    assert requirement.marker is None and requirement.url is None, str(path)
    specifiers = list(requirement.specifier)
    assert len(specifiers) == 1, f"{path} must use one exact protobuf pin"
    specifier = specifiers[0]
    assert specifier.operator == "==" and "*" not in specifier.version, str(path)
    version = Version(specifier.version)
    assert not version.is_prerelease and not version.is_devrelease, str(path)
    assert Version("6.33.6") <= version < Version("7.0.0"), f"{path} pins {version}"


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
@pytest.mark.skipif(grpcclient is None, reason=TRITON_SKIP_REASON)
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


def _load_with_triton_error(error: Exception) -> dict[str, Any]:
    original_import = builtins.__import__

    def import_with_error(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "tritonclient.grpc":
            raise error
        return original_import(name, *args, **kwargs)

    with patch.object(builtins, "__import__", side_effect=import_with_error):
        return runpy.run_path(__file__)


@pytest.mark.unit
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.parallel
class TestDependencyGuards:
    def test_triton_example_pins_match_frontend(self) -> None:
        root = Path(__file__).resolve().parents[3]
        frontend = (root / "container/deps/requirements.frontend.txt").read_text()
        dockerfile = (root / "examples/backends/tritonserver/Dockerfile").read_text()
        _assert_triton_pins_match(frontend, dockerfile)

    @pytest.mark.parametrize("package", ["tritonclient", "protobuf", "grpcio"])
    @pytest.mark.parametrize("mutation", ["changed", "missing", "duplicate"])
    def test_triton_example_pin_drift(self, package: str, mutation: str) -> None:
        frontend = (
            "tritonclient[grpc]==2.72.0\nprotobuf==6.33.6\ngrpcio>=1.81.1,<=1.83.1"
        )
        requirements = frontend.splitlines()
        pin = next(value for value in requirements if value.startswith(package))
        replacement = {
            "changed": f"{package}==1.0.0",
            "missing": "",
            "duplicate": f"'{pin}' '{pin}'",
        }[mutation]
        dockerfile = "RUN uv pip install " + " ".join(f"'{p}'" for p in requirements)
        _assert_triton_pins_match(frontend, dockerfile)
        with pytest.raises(AssertionError):
            _assert_triton_pins_match(
                frontend, dockerfile.replace(f"'{pin}'", replacement)
            )

    @pytest.mark.parametrize(
        "pin",
        [
            "protobuf==6.33.6",
            "protobuf==6.33.7",
            "  protobuf==6.34.0  # patched runtime",
        ],
    )
    def test_patched_protobuf_pins(self, pin: str) -> None:
        with patch.object(Path, "read_text", return_value=pin):
            test_protobuf_requirements_exclude_vulnerable_versions("frontend")

    @pytest.mark.parametrize(
        "pin",
        [
            "protobuf==3.20.3",
            "protobuf==5.29.5",
            "protobuf==6.33.4",
            "protobuf==6.33.5",
            "protobuf==7.0.0",
            "protobuf==7.0.0rc1",
            "protobuf==6.33.*",
            "protobuf>=6.33.6",
            "protobuf==6.33.6; python_version >= '3.12'",
            "protobuf==6.33.6\nprotobuf==6.33.7",
        ],
    )
    def test_unsafe_or_nonexact_protobuf_pins(self, pin: str) -> None:
        with patch.object(Path, "read_text", return_value=pin):
            with pytest.raises(AssertionError):
                test_protobuf_requirements_exclude_vulnerable_versions("frontend")

    def test_missing_protobuf_pin(self) -> None:
        with patch.object(Path, "read_text", return_value="# no protobuf pin"):
            with pytest.raises(AssertionError, match="Expected one protobuf"):
                test_protobuf_requirements_exclude_vulnerable_versions("frontend")

    @pytest.mark.parametrize(
        "error",
        [
            ModuleNotFoundError("No module named 'tritonclient'"),
            RuntimeError(
                "The grpc package installed is at version 1.76.0, but the generated "
                "code in grpc_service_pb2_grpc.py depends on grpcio>=1.81.1. "
                "Please upgrade your grpc module to grpcio>=1.81.1."
            ),
        ],
    )
    def test_unavailable_triton_preserves_requirement_tests(
        self, error: Exception
    ) -> None:
        namespace = _load_with_triton_error(error)
        assert namespace["grpcclient"] is None
        assert namespace["service_pb2"] is None
        guard = namespace["test_protobuf_requirements_exclude_vulnerable_versions"]
        for component in ("common", "frontend", "planner"):
            guard(component)
        for name in ("test_triton_protobuf_json_roundtrip", "test_request_parameters"):
            marks = namespace[name].pytestmark
            assert any(mark.name == "skipif" and mark.args[0] for mark in marks)

    def test_unrelated_triton_runtime_error_is_not_hidden(self) -> None:
        with pytest.raises(RuntimeError, match="unexpected initialization failure"):
            _load_with_triton_error(RuntimeError("unexpected initialization failure"))


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
@pytest.mark.skipif(grpcclient is None, reason=TRITON_SKIP_REASON)
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
