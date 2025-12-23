# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import tritonclient.grpc as grpcclient


class TritonEchoClient:
    """Thin, per-instance Triton gRPC client wrapper used by frontend gRPC tests.

    Why this exists:
    - Some tests run under pytest-xdist or in threaded contexts.
    - Mutating module globals (like GRPC_PORT) is not thread-safe and can cause
      cross-test contamination.
    """

    def __init__(self, *, grpc_host: str = "localhost", grpc_port: int = 8000):
        self._grpc_host = grpc_host
        self._grpc_port = int(grpc_port)

    def _server_url(self) -> str:
        return f"{self._grpc_host}:{self._grpc_port}"

    def _client(self) -> grpcclient.InferenceServerClient:
        return grpcclient.InferenceServerClient(url=self._server_url())

    def check_health(self) -> None:
        triton_client = self._client()
        assert triton_client.is_server_live()
        assert triton_client.is_server_ready()
        assert triton_client.is_model_ready("echo")

    def run_infer(self) -> None:
        triton_client = self._client()
        model_name = "echo"

        inputs = [
            grpcclient.InferInput("INPUT0", [16], "INT32"),
            grpcclient.InferInput("INPUT1", [16], "BYTES"),
        ]

        input0_data = np.arange(start=0, stop=16, dtype=np.int32).reshape([16])
        input1_data = np.array(
            [str(x).encode("utf-8") for x in input0_data.reshape(input0_data.size)],
            dtype=np.object_,
        ).reshape([16])

        inputs[0].set_data_from_numpy(input0_data)
        inputs[1].set_data_from_numpy(input1_data)

        results = triton_client.infer(model_name=model_name, inputs=inputs)

        output0_data = results.as_numpy("INPUT0")
        output1_data = results.as_numpy("INPUT1")

        assert (
            output0_data is not None
        ), "Expected response to include output tensor 'INPUT0'"
        assert (
            output1_data is not None
        ), "Expected response to include output tensor 'INPUT1'"
        assert np.array_equal(input0_data, output0_data)
        assert np.array_equal(input1_data, output1_data)

    def get_config(self) -> None:
        triton_client = self._client()
        model_name = "echo"
        response = triton_client.get_model_config(model_name=model_name)
        # Check one of the field that can only be set by providing Triton model config
        assert response.config.model_transaction_policy.decoupled
