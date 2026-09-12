# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the dynamo.triton worker entry point (main.py):
_register_and_serve wires the RequestHandler into a slugified endpoint and
registers the tensor model with the Dynamo runtime."""

import asyncio
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.triton import main

pytestmark = [
    pytest.mark.unit,
    pytest.mark.triton,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def patched_worker(monkeypatch):
    """Patch the worker's registration collaborators with mocks."""
    register_model = AsyncMock(name="register_model")
    monkeypatch.setattr(main, "text_format", MagicMock(name="text_format"))
    monkeypatch.setattr(main, "register_model", register_model)
    return types.SimpleNamespace(register_model=register_model)


def test_register_and_serve_registers_and_serves(patched_worker, tmp_path):
    """The registration path slugifies the endpoint, registers the tensor model,
    and serves RequestHandler.generate bound to the loaded model."""
    model_name = "identity"
    (tmp_path / model_name).mkdir()
    (tmp_path / model_name / "config.pbtxt").write_text('name: "identity"\n')

    endpoint = MagicMock(name="endpoint")
    endpoint.serve_endpoint = AsyncMock()
    runtime = MagicMock(name="runtime")
    runtime.endpoint.return_value = endpoint
    config = MagicMock(name="config")
    config.namespace = "dynamo"
    config.server_id = "triton"

    loaded_model = MagicMock(name="model")
    server = MagicMock(name="server")
    server.model.return_value = loaded_model

    asyncio.run(
        main._register_and_serve(runtime, config, server, str(tmp_path), model_name)
    )

    expected_path = (
        f"{config.namespace}.{config.server_id}.{main.endpoint_slug(model_name)}"
    )
    runtime.endpoint.assert_called_once_with(expected_path)
    server.model.assert_called_once_with(model_name)

    patched_worker.register_model.assert_awaited_once()
    reg_args, reg_kwargs = patched_worker.register_model.call_args
    assert reg_args[0] == main.ModelInput.Tensor
    assert reg_args[1] == main.ModelType.TensorBased
    assert reg_args[2] is endpoint
    assert reg_args[3] == model_name
    assert reg_kwargs["worker_type"] == main.WorkerType.Aggregated

    # register_model receives the tensor protocol layout via tensor_model_config.
    sent_config = reg_kwargs["tensor_model_config"]
    assert sent_config["name"] == ""
    assert sent_config["inputs"] == []
    assert sent_config["outputs"] == []
    assert "triton_model_config" in sent_config

    endpoint.serve_endpoint.assert_awaited_once()
    served = endpoint.serve_endpoint.call_args.args[0]
    # The served callable is RequestHandler.generate bound to the loaded model.
    assert served.__name__ == "generate"
    assert served.__self__._model is loaded_model
    assert served.__self__._server is server

    # A Triton-specific health check payload is registered with the endpoint so
    # framework probes emit backend-specific labels and carry the canary marker.
    served_kwargs = endpoint.serve_endpoint.call_args.kwargs
    health_check_payload = served_kwargs["health_check_payload"]
    assert health_check_payload["model"] == model_name
    assert health_check_payload[HEALTH_CHECK_KEY] is True


def test_register_and_serve_missing_model_error(patched_worker, tmp_path):
    """A missing config.pbtxt surfaces as FileNotFoundError before registration."""
    runtime = MagicMock(name="runtime")
    server = MagicMock(name="server")
    config = MagicMock(name="config")
    config.namespace = "dynamo"
    config.server_id = "triton"

    with pytest.raises(FileNotFoundError):
        asyncio.run(
            main._register_and_serve(
                runtime, config, server, str(tmp_path), "absent_model"
            )
        )

    patched_worker.register_model.assert_not_awaited()
