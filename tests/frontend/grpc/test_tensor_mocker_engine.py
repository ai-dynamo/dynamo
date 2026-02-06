# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parallelization: Hermetic test (xdist-safe via dynamic ports).
# Tested on: Linux (Ubuntu 24.04 container), Intel(R) Core(TM) i9-14900K, 32 vCPU.
# Combined pre_merge wall time (this file + test_tensor_parameters.py):
# - Serialized: 87.48s.
# - Parallel (-n auto): 25.27s (62.21s saved, 3.46x).
# GPU Requirement: gpu_0 (CPU-only, echo worker does not use GPU)

"""gRPC tensor echo test with mocker worker."""

from __future__ import annotations

import logging
import os
import queue
import shutil
from functools import partial

import numpy as np
import pytest
import triton_echo_client
import tritonclient.grpc as grpcclient

from tests.utils.constants import QWEN
from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN


class MockWorkerProcess(ManagedProcess):
    def __init__(self, request, system_port: int, worker_id: str = "mocker-worker"):
        self.worker_id = worker_id
        self.system_port = system_port

        command = [
            "python3",
            os.path.join(os.path.dirname(__file__), "echo_tensor_worker.py"),
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(system_port)

        log_dir = f"{request.node.name}_{worker_id}"

        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                # gRPC doesn't expose endpoint for listing models, so skip this check
                # (f"http://localhost:{grpc_port}/v1/models", check_models_api),
                (f"http://localhost:{system_port}/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=[],
            straggler_commands=["echo_tensor_worker.py"],
            log_dir=log_dir,
        )

    def is_ready(self, response) -> bool:
        try:
            status = (response.json() or {}).get("status")
        except ValueError:
            logger.warning("%s health response is not valid JSON", self.worker_id)
            return False

        is_ready = status == "ready"
        if is_ready:
            logger.info("%s status is ready", self.worker_id)
        else:
            logger.warning("%s status is not ready: %s", self.worker_id, status)
        return is_ready


@pytest.fixture(scope="function")
def start_services_with_echo_worker(request, start_services_with_grpc):
    """Start echo worker with the shared gRPC frontend.

    Function-scoped to allow parallel test execution.
    Each test gets its own gRPC frontend + echo worker on unique ports.
    No namespace conflicts because runtime_services_dynamic_ports provides isolated Etcd/NATS.
    """
    frontend_port, system_port = start_services_with_grpc
    with MockWorkerProcess(request, system_port):
        logger.info(f"gRPC Echo Worker started for test on port {frontend_port}")
        yield frontend_port


@pytest.mark.pre_merge
@pytest.mark.gpu_0  # Echo worker is CPU-only (no GPU required)
@pytest.mark.parallel
@pytest.mark.integration
@pytest.mark.model(TEST_MODEL)
def test_echo(start_services_with_echo_worker) -> None:
    frontend_port = start_services_with_echo_worker
    # Use a per-test client instance to avoid cross-test/global state issues.
    client = triton_echo_client.TritonEchoClient(grpc_port=frontend_port)
    client.check_health()
    client.run_infer()
    client.run_stream_infer()
    client.get_config()


@pytest.mark.e2e
@pytest.mark.pre_merge
@pytest.mark.gpu_0  # Echo tensor worker is CPU-only (no GPU required)
@pytest.mark.parallel
@pytest.mark.parametrize(
    "request_params",
    [
        {"malformed_response": True},
        {"raise_exception": True},
    ],
    ids=["malformed_response", "raise_exception"],
)
def test_model_infer_failure(start_services_with_echo_worker, request_params):
    """Test gRPC request-level parameters are echoed through tensor models.

    The worker acts as an identity function: echoes input tensors unchanged and
    returns all request parameters plus a "processed" flag to verify the complete
    parameter flow through the gRPC frontend.
    """
    frontend_port = start_services_with_echo_worker
    client = grpcclient.InferenceServerClient(f"localhost:{frontend_port}")

    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    inputs = [grpcclient.InferInput("INPUT", input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)

    # expect exception during inference
    with pytest.raises(Exception) as excinfo:
        client.infer("echo", inputs=inputs, parameters=request_params)
    if "malformed_response" in request_params:
        assert "missing field `data_type`" in str(excinfo.value).lower()
    elif "raise_exception" in request_params:
        assert "intentional exception" in str(excinfo.value).lower()


@pytest.mark.e2e
@pytest.mark.pre_merge
@pytest.mark.gpu_0  # Echo tensor worker is CPU-only (no GPU required)
@pytest.mark.parallel
@pytest.mark.parametrize(
    "request_params",
    [
        {"malformed_response": True},
        {"raise_exception": True},
        {"data_mismatch": True},
    ],
    ids=["malformed_response", "raise_exception", "data_mismatch"],
)
def test_model_stream_infer_failure(start_services_with_echo_worker, request_params):
    """Test gRPC request-level parameters are echoed through tensor models.

    The worker acts as an identity function: echoes input tensors unchanged and
    returns all request parameters plus a "processed" flag to verify the complete
    parameter flow through the gRPC frontend.
    """
    frontend_port = start_services_with_echo_worker
    client = grpcclient.InferenceServerClient(f"localhost:{frontend_port}")

    input_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    inputs = [grpcclient.InferInput("INPUT", input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)

    class UserData:
        def __init__(self):
            self._completed_requests: queue.Queue[
                grpcclient.InferResult | Exception
            ] = queue.Queue()

    # Define the callback function. Note the last two parameters should be
    # result and error. InferenceServerClient would povide the results of an
    # inference as grpcclient.InferResult in result. For successful
    # inference, error will be None, otherwise it will be an object of
    # tritonclientutils.InferenceServerException holding the error details
    def callback(user_data, result, error):
        print("Received callback")
        if error:
            user_data._completed_requests.put(error)
        else:
            user_data._completed_requests.put(result)

    user_data = UserData()
    client.start_stream(
        callback=partial(callback, user_data),
    )

    client.async_stream_infer(
        model_name="echo",
        inputs=inputs,
        parameters=request_params,
    )

    # For stream infer, the exception and error will pass to the callback but not
    # raised
    with pytest.raises(Exception) as excinfo:
        data_item = user_data._completed_requests.get(timeout=5)
        if isinstance(data_item, Exception):
            print("Raising exception received from stream infer callback")
            raise data_item
    if "malformed_response" in request_params:
        assert "missing field `data_type`" in str(excinfo.value).lower()
    elif "data_mismatch" in request_params:
        assert "shape implies" in str(excinfo.value).lower()
    elif "raise_exception" in request_params:
        assert "intentional exception" in str(excinfo.value).lower()
