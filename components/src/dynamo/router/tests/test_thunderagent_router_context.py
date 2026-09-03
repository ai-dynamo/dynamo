# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for trace context forwarding in the custom Python router."""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def stub_module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def load_thunderagent_router_handler():
    placeholder_type = type("Placeholder", (), {})
    stubs = {
        "uvloop": stub_module("uvloop", run=lambda coroutine: coroutine),
        "dynamo": stub_module("dynamo"),
        "dynamo.llm": stub_module(
            "dynamo.llm",
            KvRouter=placeholder_type,
            ModelInput=placeholder_type,
            ModelRuntimeConfig=placeholder_type,
            ModelType=placeholder_type,
            WorkerType=placeholder_type,
            register_model=lambda **kwargs: None,
        ),
        "dynamo.runtime": stub_module(
            "dynamo.runtime",
            DistributedRuntime=placeholder_type,
            dynamo_worker=lambda: lambda function: function,
        ),
        "dynamo.runtime.logging": stub_module(
            "dynamo.runtime.logging", configure_dynamo_logging=lambda: None
        ),
        "dynamo.thunderagent_router": stub_module("dynamo.thunderagent_router"),
        "dynamo.thunderagent_router.args": stub_module(
            "dynamo.thunderagent_router.args",
            ThunderAgentRouterConfig=placeholder_type,
            build_aic_perf_config=lambda config: config,
            build_kv_router_config=lambda config: config,
            parse_args=lambda argv=None: argv,
        ),
        "dynamo.thunderagent_router.capacity": stub_module(
            "dynamo.thunderagent_router.capacity",
            WorkerCapacityProvider=placeholder_type,
        ),
        "dynamo.thunderagent_router.program_state": stub_module(
            "dynamo.thunderagent_router.program_state",
            ReplicaKey=tuple[int, int],
        ),
        "dynamo.thunderagent_router.router": stub_module(
            "dynamo.thunderagent_router.router",
            ThunderAgentScheduler=placeholder_type,
        ),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        module_path = Path(__file__).parents[2] / "thunderagent_router" / "__main__.py"
        spec = importlib.util.spec_from_file_location(
            "thunderagent_router_main", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.ThunderAgentRouterHandler
    finally:
        for name, previous_module in previous.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


ThunderAgentRouterHandler = load_thunderagent_router_handler()


@pytest.mark.asyncio
async def test_passthrough_forwards_context_to_kv_router() -> None:
    handler = ThunderAgentRouterHandler.__new__(ThunderAgentRouterHandler)
    handler._scheduler = object()
    handler._kv_router = AsyncMock()
    handler._stat_requests_total = 0
    handler._stat_program_requests = 0
    handler._stat_passthrough_requests = 0
    handler._stat_session_final_requests = 0

    context = object()

    async def responses():
        yield {"token_ids": [4], "finish_reason": "stop"}

    handler._kv_router.generate_from_request.return_value = responses()
    request = {"model": "test-model", "token_ids": [1, 2, 3]}

    results = [output async for output in handler.generate(request, context=context)]

    assert results == [{"token_ids": [4], "finish_reason": "stop"}]
    handler._kv_router.generate_from_request.assert_awaited_once()
    _, kwargs = handler._kv_router.generate_from_request.await_args
    assert kwargs == {"context": context}
