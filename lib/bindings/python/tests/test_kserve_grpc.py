# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
import queue
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Tuple

import pytest

try:
    import tritonclient.grpc.model_config_pb2 as mc
    from tritonclient.utils import InferenceServerException
except ImportError:
    mc = None
    InferenceServerException = None

from dynamo.llm import KserveGrpcService, ModelRuntimeConfig, PythonAsyncEngine

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


async def _fetch_model_config(
    client,
    model_name: str,
    retries: int = 30,
) -> Any:
    last_error: Optional[Exception] = None
    for _ in range(retries):
        try:
            return await asyncio.to_thread(client.get_model_config, model_name)
        except InferenceServerException as err:
            last_error = err
            await asyncio.sleep(0.1)
    raise AssertionError(
        f"Unable to fetch model config for '{model_name}': {last_error}"
    )


class EchoTensorEngine:
    """Minimal tensor engine stub for registering tensor models."""

    def __init__(self, model_name: str):
        self._model_name = model_name

    def generate(self, request, context=None):
        async def _generator():
            response = {
                "model": self._model_name,
                "tensors": request.get("tensors", []),
                "parameters": request.get("parameters", {}),
            }
            if request.get("parameters", {}).get("reused_mutable"):
                for sequence in range(64):
                    response["model"] = f"{self._model_name}-{sequence}"
                    yield response
                return

            yield response

        return _generator()


@pytest.fixture
def tensor_service(runtime):
    @asynccontextmanager
    async def _start(
        model_name: str,
        *,
        runtime_config: Optional[ModelRuntimeConfig] = None,
        tensor_model_config: Optional[dict[str, Any]] = None,
        checksum: str = "dummy-mdcsum",
    ) -> AsyncIterator[Tuple[str, int]]:
        host = "127.0.0.1"
        port = 8787
        loop = asyncio.get_running_loop()
        engine = PythonAsyncEngine(EchoTensorEngine(model_name).generate, loop)
        tensor_model_service = KserveGrpcService(port=port, host=host)

        tensor_model_service.add_tensor_model(
            model_name,
            checksum,
            engine,
            runtime_config=runtime_config,
            tensor_model_config=tensor_model_config,
        )

        async def _serve():
            await tensor_model_service.run(runtime)

        server_task = asyncio.create_task(_serve())
        try:
            await asyncio.sleep(1)  # wait service to start
            yield host, port
        finally:
            tensor_model_service.shutdown()
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(server_task, timeout=5)

    return _start


@pytest.mark.asyncio
@pytest.mark.forked
async def test_model_config_uses_tensor_model_config(tensor_service):
    """Ensure tensor metadata is returned via the ModelConfig endpoint."""
    import tritonclient.grpc as grpcclient

    model_name = "tensor-config-model"
    tensor_config = {
        "name": model_name,
        "inputs": [
            {"name": "input_text", "data_type": "Bytes", "shape": [-1]},
            {"name": "control_flag", "data_type": "Bool", "shape": [1]},
        ],
        "outputs": [
            {"name": "results", "data_type": "Bytes", "shape": [-1]},
        ],
    }
    async with tensor_service(model_name, tensor_model_config=tensor_config) as (
        host,
        port,
    ):
        client = grpcclient.InferenceServerClient(url=f"{host}:{port}")
        try:
            response = await _fetch_model_config(client, model_name)
        finally:
            client.close()

    model_config = response.config
    assert model_config.name == model_name
    assert model_config.platform == "dynamo"
    assert model_config.backend == "dynamo"

    inputs = {spec.name: spec for spec in model_config.input}
    assert list(inputs["input_text"].dims) == [-1]
    assert inputs["input_text"].data_type == mc.TYPE_STRING
    assert list(inputs["control_flag"].dims) == [1]
    assert inputs["control_flag"].data_type == mc.TYPE_BOOL

    outputs = {spec.name: spec for spec in model_config.output}
    assert list(outputs["results"].dims) == [-1]
    assert outputs["results"].data_type == mc.TYPE_STRING


@pytest.mark.asyncio
@pytest.mark.forked
async def test_model_config_missing_tensor_config_errors(tensor_service):
    """ModelConfig should return NOT_FOUND when no tensor metadata is saved."""
    model_name = "tensor-config-missing"
    import tritonclient.grpc as grpcclient

    async with tensor_service(model_name) as (host, port):
        client = grpcclient.InferenceServerClient(url=f"{host}:{port}")
        try:
            with pytest.raises(InferenceServerException) as excinfo:
                await asyncio.to_thread(client.get_model_config, model_name)
        finally:
            client.close()

    assert "not found" in str(excinfo.value).lower()


@pytest.mark.asyncio
@pytest.mark.forked
@pytest.mark.timeout(30)
async def test_python_async_engine_snapshots_reused_mutable_responses(tensor_service):
    """Snapshot each typed response before polling a reused Python object again."""
    import numpy as np
    import tritonclient.grpc as grpcclient

    model_name = "tensor-reused-mutable"
    async with tensor_service(model_name) as (host, port):
        client = grpcclient.InferenceServerClient(url=f"{host}:{port}")
        completed: queue.Queue = queue.Queue()

        def callback(result, error):
            completed.put(error if error is not None else result)

        input_data = np.array([1], dtype=np.int32)
        infer_input = grpcclient.InferInput("INPUT0", input_data.shape, "INT32")
        infer_input.set_data_from_numpy(input_data)

        client.start_stream(callback=callback)
        try:
            client.async_stream_infer(
                model_name=model_name,
                inputs=[infer_input],
                parameters={"reused_mutable": True},
            )
            responses = [
                await asyncio.to_thread(completed.get, True, 5) for _ in range(64)
            ]
        finally:
            client.stop_stream()
            client.close()

    errors = [response for response in responses if isinstance(response, Exception)]
    assert not errors
    assert [response.get_response().model_name for response in responses] == [
        f"{model_name}-{sequence}" for sequence in range(64)
    ]
