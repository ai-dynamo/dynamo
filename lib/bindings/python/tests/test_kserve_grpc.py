# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional, Tuple

import pytest
import tritonclient.grpc.model_config_pb2 as mc
from tritonclient.utils import InferenceServerException

from dynamo.llm import KserveGrpcService, ModelRuntimeConfig, PythonAsyncEngine

pytestmark = pytest.mark.pre_merge


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
            yield {
                "model": self._model_name,
                "tensors": request.get("tensors", []),
                "parameters": request.get("parameters", {}),
            }

        return _generator()


@pytest.fixture
def tensor_service(runtime):
    @asynccontextmanager
    async def _start(
        model_name: str,
        *,
        runtime_config: Optional[ModelRuntimeConfig] = None,
        checksum: str = "dummy-mdcsum",
    ) -> AsyncIterator[Tuple[str, int]]:
        host = "127.0.0.1"
        port = 8787
        loop = asyncio.get_running_loop()
        engine = PythonAsyncEngine(EchoTensorEngine(model_name).generate, loop)
        tensor_model_service = KserveGrpcService(port=port, host=host)

        tensor_model_service.add_tensor_model(
            model_name, checksum, engine, runtime_config=runtime_config
        )

        cancel_token = runtime.child_token()

        async def _serve():
            await tensor_model_service.run(cancel_token)

        server_task = asyncio.create_task(_serve())
        try:
            await asyncio.sleep(1)  # wait service to start
            yield host, port
        finally:
            cancel_token.cancel()
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(server_task, timeout=5)

    return _start


@pytest.mark.asyncio
@pytest.mark.forked
async def test_model_config_uses_runtime_config(tensor_service):
    """Ensure tensor runtime_config is returned via the ModelConfig endpoint."""
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
    runtime_config = ModelRuntimeConfig()
    runtime_config.set_tensor_model_config(tensor_config)

    async with tensor_service(model_name, runtime_config=runtime_config) as (
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
async def test_model_config_missing_runtime_config_errors(tensor_service):
    """ModelConfig should return NOT_FOUND when no tensor runtime_config is saved."""
    model_name = "tensor-config-missing"
    import tritonclient.grpc as grpcclient

    async with tensor_service(model_name, runtime_config=None) as (host, port):
        client = grpcclient.InferenceServerClient(url=f"{host}:{port}")
        try:
            with pytest.raises(InferenceServerException) as excinfo:
                await asyncio.to_thread(client.get_model_config, model_name)
        finally:
            client.close()

    assert "not found" in str(excinfo.value).lower()
