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


def stream(*chunks):
    async def generator():
        for chunk in chunks:
            yield chunk

    return generator()


@pytest.mark.asyncio
async def test_generate_forwards_request_fields_it_does_not_use() -> None:
    handler, router = handler_with_router()
    router.generate_from_request.return_value = stream()
    request = {
        "model": "qwen-vl",
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"image_url": [{"Url": "https://example.com/a.png"}]},
        "multi_modal_uuids": {"image_url": ["img-0"]},
        "media_io_kwargs": {"image": {"num_frames": 1}},
        "kv_hint": {"cache_namespace": "tenant-a"},
        "agent_context": {"program_id": "p-1"},
        "require_reasoning": True,
        "request_timestamp_ms": 1.5,
    }

    _ = [chunk async for chunk in handler.generate(request)]

    forwarded = router.generate_from_request.await_args.args[0]
    for key, value in request.items():
        assert forwarded[key] == value, key


@pytest.mark.asyncio
async def test_generate_keeps_defaults_and_legacy_dp_rank() -> None:
    handler, router = handler_with_router()
    router.generate_from_request.return_value = stream()

    _ = [chunk async for chunk in handler.generate({"token_ids": [1], "dp_rank": 3})]

    forwarded = router.generate_from_request.await_args.args[0]
    assert forwarded["model"] == "unknown"
    assert forwarded["stop_conditions"] == {}
    assert forwarded["sampling_options"] == {}
    assert forwarded["output_options"] == {}
    assert forwarded["eos_token_ids"] == []
    assert forwarded["annotations"] == []
    assert forwarded["routing"] == {"dp_rank": 3}


@pytest.mark.asyncio
async def test_generate_does_not_override_explicit_routing_with_dp_rank() -> None:
    handler, router = handler_with_router()
    router.generate_from_request.return_value = stream()
    routing = {"dp_rank": 1, "backend_instance_id": 9}

    _ = [
        chunk
        async for chunk in handler.generate(
            {"token_ids": [1], "routing": routing, "dp_rank": 3}
        )
    ]

    assert router.generate_from_request.await_args.args[0]["routing"] == routing


@pytest.mark.asyncio
async def test_generate_forwards_worker_output_unchanged() -> None:
    handler, router = handler_with_router()
    chunk = {
        "token_ids": [],
        "output_type": "image",
        "content_parts": [{"type": "image_url", "image_url": {"url": "data:,"}}],
        "encoder_result": {"embeddings_ref": "e-0"},
        "worker_trace_link": {"trace_id": "t-0"},
        "finish_reason": "stop",
    }
    router.generate_from_request.return_value = stream(chunk)

    outputs = [output async for output in handler.generate({"token_ids": [1]})]

    assert outputs == [chunk]
