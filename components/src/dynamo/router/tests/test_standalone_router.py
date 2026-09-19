# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def load_standalone_router_handler():
    placeholder_type = type("Placeholder", (), {})
    stubs = {
        "uvloop": stub_module("uvloop", run=lambda coroutine: coroutine),
        "dynamo": stub_module("dynamo"),
        "dynamo.llm": stub_module(
            "dynamo.llm",
            AicPerfConfig=placeholder_type,
            KvRouter=placeholder_type,
            KvRouterConfig=placeholder_type,
        ),
        "dynamo.router": stub_module("dynamo.router"),
        "dynamo.router.args": stub_module(
            "dynamo.router.args",
            DynamoRouterConfig=placeholder_type,
            build_aic_perf_config=lambda config: config,
            build_kv_router_config=lambda config: config,
            parse_args=lambda argv=None: argv,
        ),
        "dynamo.runtime": stub_module(
            "dynamo.runtime",
            Client=placeholder_type,
            DistributedRuntime=placeholder_type,
            dynamo_worker=lambda: lambda function: function,
        ),
        "dynamo.runtime.logging": stub_module(
            "dynamo.runtime.logging", configure_dynamo_logging=lambda: None
        ),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        module_path = Path(__file__).parents[1] / "__main__.py"
        spec = importlib.util.spec_from_file_location(
            "standalone_router_main", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.StandaloneRouterHandler
    finally:
        for name, previous_module in previous.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


StandaloneRouterHandler = load_standalone_router_handler()


def handler_with_router():
    handler = StandaloneRouterHandler.__new__(StandaloneRouterHandler)
    router = AsyncMock()
    handler.kv_router = router
    return handler, router


@pytest.mark.asyncio
async def test_best_worker_id_forwards_cache_namespace() -> None:
    handler, router = handler_with_router()
    router.best_worker.return_value = (7, 0, 3)

    results = [
        worker_id
        async for worker_id in handler.best_worker_id(
            [1, 2, 3, 4],
            {"temperature": 0.0},
            cache_namespace="tenant-a",
        )
    ]

    assert results == [7]
    router.best_worker.assert_awaited_once_with(
        [1, 2, 3, 4],
        {"temperature": 0.0},
        cache_namespace="tenant-a",
    )


@pytest.mark.asyncio
async def test_get_overlap_scores_forwards_cache_namespace() -> None:
    handler, router = handler_with_router()
    router.get_overlap_scores.return_value = {"workers": []}
    request = {
        "token_ids": [1, 2, 3, 4],
        "router_config_override": {"temperature": 0.0},
        "block_mm_infos": None,
        "lora_name": "adapter-a",
        "include_shared": False,
        "cache_namespace": "tenant-a",
    }

    results = [scores async for scores in handler.get_overlap_scores(request)]

    assert results == [{"workers": []}]
    router.get_overlap_scores.assert_awaited_once_with(
        [1, 2, 3, 4],
        {"temperature": 0.0},
        None,
        "adapter-a",
        False,
        "tenant-a",
    )


@pytest.mark.asyncio
async def test_generate_forwards_fields_added_after_the_wrapper() -> None:
    """Fields the wrapper does not name must still reach the kv router.

    PreprocessedRequest grows fields after this router ships (e.g.
    require_reasoning); rebuilding the request from an explicit allow-list
    dropped them, so the backend saw builder defaults -- in particular
    require_reasoning=false, which turns off reasoning token accounting.
    """
    handler, router = handler_with_router()
    request = {
        "model": "test-model",
        "token_ids": [1, 2, 3],
        "require_reasoning": True,
        "kv_hint": {"block_ids": [7]},
        "router": {"ttft_target": 42},
        "agent_context": {"user_id": "u1"},
        "media_io_kwargs": {"allow_remote": True},
        "is_probe": True,
        "request_timestamp_ms": 123.5,
        "migration_link": {"trace_id": "t", "span_id": "s"},
    }

    async def responses():
        yield {"token_ids": [4], "finish_reason": "stop"}

    router.generate_from_request.return_value = responses()

    _ = [output async for output in handler.generate(request)]

    forwarded = router.generate_from_request.await_args.args[0]
    assert forwarded["require_reasoning"] is True
    assert forwarded["kv_hint"] == {"block_ids": [7]}
    assert forwarded["router"] == {"ttft_target": 42}
    assert forwarded["agent_context"] == {"user_id": "u1"}
    assert forwarded["media_io_kwargs"] == {"allow_remote": True}
    assert forwarded["is_probe"] is True
    assert forwarded["request_timestamp_ms"] == 123.5
    assert forwarded["migration_link"] == {"trace_id": "t", "span_id": "s"}


@pytest.mark.asyncio
async def test_generate_synthesizes_routing_from_legacy_dp_rank() -> None:
    handler, router = handler_with_router()
    request = {"model": "test-model", "token_ids": [1, 2, 3], "dp_rank": 3}

    async def responses():
        yield {"token_ids": [4], "finish_reason": "stop"}

    router.generate_from_request.return_value = responses()

    _ = [output async for output in handler.generate(request)]

    forwarded = router.generate_from_request.await_args.args[0]
    assert forwarded["routing"] == {"dp_rank": 3}
    assert forwarded["model"] == "test-model"
    assert forwarded["token_ids"] == [1, 2, 3]


@pytest.mark.asyncio
async def test_generate_applies_model_fallback_when_missing() -> None:
    handler, router = handler_with_router()
    request = {"token_ids": [9]}

    async def responses():
        yield {"token_ids": [4], "finish_reason": "stop"}

    router.generate_from_request.return_value = responses()

    _ = [output async for output in handler.generate(request)]

    forwarded = router.generate_from_request.await_args.args[0]
    assert forwarded["model"] == "unknown"
    assert forwarded["routing"] is None
    assert forwarded["token_ids"] == [9]
