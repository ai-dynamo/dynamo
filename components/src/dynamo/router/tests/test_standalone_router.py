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


def load_token_ids_module():
    module_path = Path(__file__).parents[2] / "common" / "utils" / "token_ids.py"
    spec = importlib.util.spec_from_file_location(
        "dynamo.common.utils.token_ids", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_standalone_router_handler():
    placeholder_type = type("Placeholder", (), {})
    stubs = {
        "uvloop": stub_module("uvloop", run=lambda coroutine: coroutine),
        "dynamo": stub_module("dynamo"),
        "dynamo.common": stub_module("dynamo.common"),
        "dynamo.common.utils": stub_module("dynamo.common.utils"),
        "dynamo.common.utils.token_ids": load_token_ids_module(),
        "dynamo.llm": stub_module(
            "dynamo.llm",
            AisPerfConfig=placeholder_type,
            KvRouter=placeholder_type,
            KvRouterConfig=placeholder_type,
        ),
        "dynamo.router": stub_module("dynamo.router"),
        "dynamo.router.args": stub_module(
            "dynamo.router.args",
            DynamoRouterConfig=placeholder_type,
            build_ais_perf_config=lambda config: config,
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
async def test_generate_forwards_request_and_response_fields() -> None:
    handler, router = handler_with_router()
    worker_output = {
        "token_ids": [5],
        "output_type": "image",
        "content_parts": [{"type": "text", "text": "hi"}],
        "worker_trace_link": {"trace_id": "t"},
    }

    async def worker_stream():
        yield worker_output

    router.generate_from_request.return_value = worker_stream()
    request = {
        "token_ids": [1, 2, 3, 4],
        "dp_rank": 2,
        "mm_routing_info": {"routing_token_ids": [9, 9]},
        "kv_hint": {"source": "prefill"},
        "agent_context": {"session_id": "s"},
    }

    results = [output async for output in handler.generate(request)]

    assert results == [worker_output]
    (forwarded,), _ = router.generate_from_request.call_args
    assert forwarded == {
        **request,
        "model": "unknown",
        "routing": {"dp_rank": 2},
    }


@pytest.mark.asyncio
async def test_generate_forwards_packed_token_ids_untouched() -> None:
    handler, router = handler_with_router()

    async def worker_stream():
        yield {"token_ids": [5]}

    router.generate_from_request.return_value = worker_stream()
    packed = b"".join(i.to_bytes(4, "little") for i in (1, 2, 3, 4))

    async for _ in handler.generate({"token_ids": packed}):
        pass

    (forwarded,), _ = router.generate_from_request.call_args
    assert forwarded["token_ids"] is packed
