# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke test: dynamo.triton serves an identity tensor through the
KServe gRPC frontend."""

import json
import logging
import os

import numpy as np
import pytest

pytest.importorskip("tritonserver")
grpcclient = pytest.importorskip("tritonclient.grpc")

from dynamo.triton.util import endpoint_slug  # noqa: E402
from tests.utils.managed_process import ManagedProcess  # noqa: E402

logger = logging.getLogger(__name__)

_MODEL_REPO = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "components",
        "src",
        "dynamo",
        "triton",
        "models",
    )
)
_MODEL_NAME = "identity"
_WORKER_TIMEOUT_SECS = 300


class TritonWorkerProcess(ManagedProcess):
    """Subprocess wrapper that boots `python3 -m dynamo.triton` against the
    bundled identity model and exposes its `/health` endpoint on `system_port`."""

    def __init__(self, request, system_port: int, log_dir: str):
        self.system_port = system_port

        command = [
            "python3",
            "-m",
            "dynamo.triton",
            "--model-repository",
            _MODEL_REPO,
            "--allow-metrics",
            "false",
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = json.dumps(
            [endpoint_slug(_MODEL_NAME)]
        )
        env["DYN_SYSTEM_PORT"] = str(system_port)

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (
                    f"http://localhost:{system_port}/health",
                    lambda r: r.json().get("status") == "ready",
                )
            ],
            timeout=_WORKER_TIMEOUT_SECS,
            display_output=True,
            log_dir=log_dir,
            terminate_all_matching_process_names=False,
        )


@pytest.fixture(scope="function")
def start_services_with_triton_worker(request, tmp_path, start_services_with_grpc):
    """Start a Triton worker alongside the shared gRPC frontend; logs land in
    the pytest-managed `tmp_path` and are cleaned up automatically."""
    frontend_port, system_port = start_services_with_grpc
    log_dir = str(tmp_path / f"{request.node.name}_worker")
    with TritonWorkerProcess(request, system_port, log_dir):
        yield frontend_port


@pytest.mark.e2e
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.triton
@pytest.mark.parallel
@pytest.mark.timeout(120)
def test_identity_tensor_round_trip(
    file_storage_backend, start_services_with_triton_worker
):
    """Send an INT32 tensor to the identity model through the KServe gRPC
    frontend and assert the response matches the input exactly."""
    frontend_port = start_services_with_triton_worker
    client = grpcclient.InferenceServerClient(f"localhost:{frontend_port}")

    # max_batch_size=1 in config.pbtxt requires the outer batch dim on the wire.
    input_data = np.array([[1, 2, 3, 4]], dtype=np.int32)
    inputs = [grpcclient.InferInput("INPUT0", input_data.shape, "INT32")]
    inputs[0].set_data_from_numpy(input_data)

    response = client.infer(_MODEL_NAME, inputs=inputs)

    output_data = response.as_numpy("OUTPUT0")
    assert output_data is not None
    assert np.array_equal(input_data, output_data)
