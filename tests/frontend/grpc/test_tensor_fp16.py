# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Parallelization: Hermetic test (xdist-safe via dynamic ports).
# Tested on: Linux (Ubuntu 24.04 container), Intel(R) Core(TM) i9-14900K, 32 vCPU.
# GPU Requirement: gpu_0 (CPU-only, echo tensor worker does not use GPU)

"""End-to-end FP16 roundtrip through the KServe gRPC tensor path.

The echo worker (echo_tensor_worker.py) declares its model with STRING/INT32
inputs, but the frontend performs no dtype validation between request and
model config (see the note by gluo in echo_tensor_worker.py:59-61). This test
uses that documented invariant to send FP16 through the shared worker without
modifying test infrastructure — if the invariant is ever tightened this test
starts failing on purpose and forces the discussion.

BF16 is intentionally not covered here because numpy has no native bfloat16
dtype and ml_dtypes is not a repo dep. The Rust unit tests in
lib/llm/src/grpc/service/tensor.rs and lib/llm/src/protocols/tensor.rs cover
BF16 encode/decode/serde end-to-end.
"""

import logging
import os
import shutil

import numpy as np
import pytest

try:
    import tritonclient.grpc as grpcclient
except ImportError:
    grpcclient = None

from tests.utils.managed_process import ManagedProcess

logger = logging.getLogger(__name__)


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

        log_dir = f"{request.node.name}_worker"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (
                    f"http://localhost:{system_port}/health",
                    lambda r: r.json().get("status") == "ready",
                )
            ],
            timeout=300,
            display_output=True,
            log_dir=log_dir,
            terminate_all_matching_process_names=False,
        )


@pytest.fixture(scope="function")
def start_services_with_echo_tensor_worker(request, start_services_with_grpc):
    frontend_port, system_port = start_services_with_grpc
    with EchoTensorWorkerProcess(request, system_port):
        logger.info(f"Echo Tensor Worker started for test on port {frontend_port}")
        yield frontend_port


@pytest.mark.e2e
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.parallel
def test_fp16_raw_path_roundtrip(
    file_storage_backend, start_services_with_echo_tensor_worker
):
    """Send FP16 via KServe raw_input_contents; assert byte-exact roundtrip.

    Values chosen for exact FP16 representability to catch sign, mantissa,
    endianness, and overflow-to-infinity bugs in one shot. 65504.0 is the
    largest finite FP16 value.
    """
    frontend_port = start_services_with_echo_tensor_worker
    client = grpcclient.InferenceServerClient(f"localhost:{frontend_port}")

    input_data = np.array(
        [1.5, -2.25, 3.125, -4.0, 0.5, 100.0, -0.0625, 65504.0], dtype=np.float16
    )
    inputs = [grpcclient.InferInput("INPUT", input_data.shape, "FP16")]
    inputs[0].set_data_from_numpy(input_data)

    response = client.infer("echo", inputs=inputs)
    output = response.as_numpy("INPUT")
    assert np.array_equal(
        input_data, output
    ), "FP16 raw-path roundtrip must be byte-exact"
    resp_msg = response.get_response()
    assert resp_msg.outputs[0].datatype == "FP16", (
        f"Frontend must advertise FP16 in response metadata, "
        f"got {resp_msg.outputs[0].datatype!r}"
    )
