# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

pytest.importorskip("torch", reason="the external encoder example requires PyTorch")
pytest.importorskip(
    "dynamo._core.backend",
    reason="dynamo._core.backend not built — run maturin develop first",
)
pytest.importorskip(
    "vllm.engine.arg_utils",
    reason="a full vLLM installation is required by the vision example",
)

from dynamo.workflow import (  # noqa: E402
    GenerateEndpointBinding,
    RemoteBinding,
    WorkflowExecutor,
)
from examples.custom_backend.user_ensemble.remote import (  # noqa: E402
    worker as remote_worker_module,
)
from examples.custom_backend.user_ensemble.remote.bindings import (  # noqa: E402
    CLASSIFIER_ENDPOINT,
    ENCODER_ENDPOINT,
    GENERATOR_ENDPOINT,
    compile_remote_workflow,
)
from examples.custom_backend.user_ensemble.remote.provider import (  # noqa: E402
    provide_workflow,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _FakeTensorCarrier:
    def __init__(self) -> None:
        self.close_calls = 0

    async def export_tensor(self, tensor: Any, transfer_id: str) -> Any:
        raise NotImplementedError

    async def export_tensor_fanout(
        self, tensor: Any, transfer_ids: tuple[str, ...]
    ) -> Any:
        raise NotImplementedError

    async def import_tensor(self, reference: Any) -> Any:
        raise NotImplementedError

    async def close(self) -> None:
        self.close_calls += 1


class _FakeEndpoint:
    def __init__(self) -> None:
        self.handler: Any = None

    async def serve_endpoint(self, handler: Any) -> None:
        self.handler = handler


class _FakeRuntime:
    def __init__(self) -> None:
        self.endpoint_ids: list[str] = []
        self.created_endpoint = _FakeEndpoint()

    def endpoint(self, endpoint_id: str) -> _FakeEndpoint:
        self.endpoint_ids.append(endpoint_id)
        return self.created_endpoint


def test_remote_plan_uses_nixl_fanout_and_stock_generate_protocol():
    plan = compile_remote_workflow()

    assert type(plan.bindings["encoder"]) is RemoteBinding
    assert type(plan.bindings["classifier"]) is RemoteBinding
    assert isinstance(plan.bindings["generator"], GenerateEndpointBinding)
    assert plan.bindings["encoder"].endpoint_id == ENCODER_ENDPOINT
    assert plan.bindings["classifier"].endpoint_id == CLASSIFIER_ENDPOINT
    assert plan.bindings["generator"].endpoint_id == GENERATOR_ENDPOINT
    assert {edge.transfer_id: edge.carrier for edge in plan.edges} == {
        "encoder.request": "inline",
        "classifier.encoder_features": "nixl",
        "generator.request": "inline",
        "generator.encoder_features": "nixl",
        "generator.encoder_metadata": "inline",
    }


async def test_frontend_provider_binds_plan_and_result_adapter(
    monkeypatch: pytest.MonkeyPatch,
):
    executor = object.__new__(WorkflowExecutor)
    bind = AsyncMock(return_value=executor)
    runtime = object()
    config = SimpleNamespace(model_path=None, model_name="served-workflow")
    template = Path("templates/vision.jinja")
    monkeypatch.setenv("DYN_MODEL", "org/text-model")
    monkeypatch.setenv("DYN_CUSTOM_JINJA_TEMPLATE", str(template))

    with patch(
        "examples.custom_backend.user_ensemble.remote.provider.WorkflowExecutor.bind",
        bind,
    ):
        application = await provide_workflow(runtime, config)

    assert bind.await_args.args == (compile_remote_workflow(),)
    assert bind.await_args.kwargs == {"runtime": runtime}
    assert application.executor is executor
    assert application.model_path == "org/text-model"
    assert application.model_name == "served-workflow"
    assert application.custom_template_path == template
    assert application.result_adapter(
        {
            "chunk": {"token_ids": [42], "engine_data": {"trace": "kept"}},
            "scores": {"answer": 1.0},
        }
    ) == {
        "token_ids": [42],
        "engine_data": {
            "trace": "kept",
            "ensemble": {"classifier_scores": {"answer": 1.0}},
        },
    }


async def test_classifier_worker_serves_workflow_protocol_and_closes_carrier():
    runtime = _FakeRuntime()
    carrier = _FakeTensorCarrier()

    with patch.object(
        remote_worker_module,
        "NixlTensorCarrier",
        return_value=carrier,
    ):
        await remote_worker_module.remote_worker.__wrapped__(
            runtime,
            "classifier",
            "unused-model",
            "unused-encoder",
        )

    assert runtime.endpoint_ids == [CLASSIFIER_ENDPOINT]
    assert runtime.created_endpoint.handler is not None
    assert carrier.close_calls == 1
