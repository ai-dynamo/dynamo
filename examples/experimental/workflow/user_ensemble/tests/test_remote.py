# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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

from dynamo.experimental.workflow import (  # noqa: E402
    GenerateEndpointBinding,
    InlineBinding,
    RemoteBinding,
    WorkflowOrchestrator,
)
from examples.experimental.workflow.user_ensemble.remote import (  # noqa: E402
    classifier_worker as classifier_worker_module,
    orchestrator_worker as orchestrator_worker_module,
)
from examples.experimental.workflow.user_ensemble.remote.bindings import (  # noqa: E402
    CLASSIFIER_ENDPOINT,
    ENCODER_ENDPOINT,
    GENERATOR_ENDPOINT,
    compile_remote_workflow,
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
    assert plan.bindings["response"] == InlineBinding("response")
    assert plan.bindings["encoder"].endpoint_id == ENCODER_ENDPOINT
    assert plan.bindings["classifier"].endpoint_id == CLASSIFIER_ENDPOINT
    assert plan.bindings["generator"].endpoint_id == GENERATOR_ENDPOINT
    assert plan.bindings["encoder"].tensor_carrier == "nixl"
    assert plan.bindings["classifier"].tensor_carrier == "nixl"
    assert plan.bindings["generator"].tensor_carrier == "nixl"
    assert not hasattr(plan, "edges")


async def test_orchestrator_worker_binds_and_registers_remote_workflow():
    orchestrator = object.__new__(WorkflowOrchestrator)
    bind = AsyncMock(return_value=orchestrator)
    register = AsyncMock()
    runtime = _FakeRuntime()
    handler = SimpleNamespace(generate=object())

    with (
        patch.object(orchestrator_worker_module.WorkflowOrchestrator, "bind", bind),
        patch.object(orchestrator_worker_module, "register_model", register),
        patch.object(
            orchestrator_worker_module,
            "WorkflowEndpointHandler",
            return_value=handler,
        ),
        patch.dict(
            orchestrator_worker_module.os.environ,
            {
                "DYN_MODEL": "org/model",
                "DYN_SERVED_MODEL_NAME": "ensemble",
                "DYN_CUSTOM_JINJA_TEMPLATE": "/tmp/template.jinja",
            },
        ),
    ):
        await orchestrator_worker_module.orchestrator_worker.__wrapped__(runtime)

    assert bind.await_args.args == (compile_remote_workflow(),)
    assert bind.await_args.kwargs["runtime"] is runtime
    response = bind.await_args.kwargs["inline_runners"]["response"]
    assert response.contract.id == "ensemble-response"
    assert runtime.endpoint_ids == [orchestrator_worker_module.ORCHESTRATOR_ENDPOINT]
    assert runtime.created_endpoint.handler is handler.generate
    assert register.await_args.args[2] is runtime.created_endpoint
    assert register.await_args.args[3] == "org/model"
    assert register.await_args.kwargs["model_name"] == "ensemble"
    assert register.await_args.kwargs["custom_template_path"] == "/tmp/template.jinja"
    assert register.await_args.kwargs["ignore_weights"] is True


async def test_classifier_worker_serves_workflow_protocol_and_closes_carrier():
    runtime = _FakeRuntime()
    carrier = _FakeTensorCarrier()

    with patch.object(
        classifier_worker_module,
        "NixlTensorCarrier",
        return_value=carrier,
    ):
        await classifier_worker_module.classifier_worker.__wrapped__(runtime)

    assert runtime.endpoint_ids == [CLASSIFIER_ENDPOINT]
    assert runtime.created_endpoint.handler is not None
    assert carrier.close_calls == 1

    def can_export(self, value: Any) -> bool:
        return False
