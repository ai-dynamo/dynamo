# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for configuration-driven SGLang engine routes."""

import argparse
import asyncio
import dataclasses
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import msgspec
import pytest

from dynamo.sglang.engine_routes import (
    EngineRouteDescriptor,
    normalize_engine_route_result,
    parse_engine_route_descriptors,
    resolve_configured_engine_routes,
)

try:
    from dynamo.sglang.backend_args import DynamoSGLangArgGroup
except ImportError:
    DynamoSGLangArgGroup = None

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


class Request:
    """Minimal stand-in for Starlette's HTTP request annotation."""


Request.__module__ = "starlette.requests"


class UpdateWeightsRequest(msgspec.Struct, kw_only=True):
    """Typed request containing the vanilla SGLang fields."""

    names: list[str]
    dtypes: list[str]
    shapes: list[list[int]]
    group_name: str = "weight_update_group"
    weight_version: str | None = None


class TypedFailure(msgspec.Struct):
    success: bool
    message: str


@dataclasses.dataclass
class PauseGenerationRequest:
    mode: str = "abort"


@dataclasses.dataclass
class ContinueGenerationRequest:
    torch_empty_cache: bool = True


class FakeTokenizerManager:
    def __init__(self):
        self.auto_create_handle_loop = Mock()
        self.update_calls = []
        self.pause_calls = []
        self.continue_calls = []

    async def flush_cache(self, timeout_s: float | None = None):
        return {"timeout_s": timeout_s}

    async def pause_generation(self, obj: PauseGenerationRequest):
        self.pause_calls.append(obj)

    async def continue_generation(self, obj: ContinueGenerationRequest):
        self.continue_calls.append(obj)

    async def update_weights_from_distributed(
        self,
        obj: UpdateWeightsRequest,
        request: Request | None = None,
    ):
        self.update_calls.append((obj, request))
        return False, f"rejected {obj.weight_version}"


class FakeEngine:
    def __init__(self, loop=None):
        self.loop = loop
        self.tokenizer_manager = FakeTokenizerManager()

    def my_custom_method(self, **kwargs):
        return {"custom": kwargs}

    async def async_custom_method(self, value):
        return {"async": value}

    async def _server_info(self):
        return {
            "tp_size": 4,
            "pp_size": 2,
            "dp_size": 1,
            "disaggregation_mode": "null",
        }

    def get_server_info(self):
        return self.loop.run_until_complete(self._server_info())

    def flush_cache(self):
        return self.loop.run_until_complete(self.tokenizer_manager.flush_cache())

    def init_weights_update_group(self, **kwargs):
        return True, kwargs["group_name"]

    destroy_weights_update_group = init_weights_update_group


def _resolved_handlers(engine, descriptors):
    return dict(resolve_configured_engine_routes(engine, descriptors))


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "my_custom_method",
            [EngineRouteDescriptor("my_custom_method", "my_custom_method", "engine")],
        ),
        (
            ["server_info=get_server_info", "pause_generation:tm"],
            [
                EngineRouteDescriptor("server_info", "get_server_info", "engine"),
                EngineRouteDescriptor("pause_generation", "pause_generation", "tm"),
            ],
        ),
        (
            "admin/update=update_weights_from_distributed:tm",
            [
                EngineRouteDescriptor(
                    "admin/update", "update_weights_from_distributed", "tm"
                )
            ],
        ),
        ("", []),
        (None, []),
    ],
)
def test_parse_engine_route_descriptors(raw, expected):
    assert parse_engine_route_descriptors(raw) == expected


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ([""], "descriptor is empty"),
        ("=method", "both the route path and method"),
        ("route=", "both the route path and method"),
        ("route=one=two", "at most one '='"),
        ("route:tm:engine", "at most one ':'"),
        ("route:worker", "unknown target"),
        ("/route=method", "route path"),
        ("route/=method", "route path"),
        ("route=bad-method", "Python identifier"),
        ("route=_private", "private methods"),
        ("route route=other", "configured more than once"),
    ],
)
def test_parse_engine_route_errors(raw, message):
    with pytest.raises(ValueError, match=message):
        parse_engine_route_descriptors(raw)


def test_repeated_cli_and_environment_configuration(monkeypatch):
    if DynamoSGLangArgGroup is None:
        pytest.skip("Dynamo runtime bindings are unavailable")

    def parse(*args):
        parser = argparse.ArgumentParser()
        DynamoSGLangArgGroup().add_arguments(parser)
        return parser.parse_args(args).engine_routes

    assert parse(
        "--engine-route",
        "server_info=get_server_info",
        "--engine-route",
        "pause_generation:tm",
    ) == ["server_info=get_server_info", "pause_generation:tm"]

    monkeypatch.setenv(
        "DYN_SGLANG_ENGINE_ROUTES",
        "flush_cache update_weights_from_distributed:tm",
    )
    assert parse() == ["flush_cache", "update_weights_from_distributed:tm"]


@pytest.mark.asyncio
async def test_compatibility_routes_resolve_and_dispatch_without_a_registry():
    engine = FakeEngine(asyncio.get_running_loop())
    routes = _resolved_handlers(
        engine,
        [
            "server_info=get_server_info",
            "flush_cache",
            "pause_generation:tm",
            "continue_generation:tm",
            "init_weights_update_group",
            "update_weights_from_distributed:tm",
            "destroy_weights_update_group",
            "async_custom=async_custom_method",
        ],
    )

    assert set(routes) == {
        "server_info",
        "flush_cache",
        "pause_generation",
        "continue_generation",
        "init_weights_update_group",
        "update_weights_from_distributed",
        "destroy_weights_update_group",
        "async_custom",
    }
    assert "call_tokenizer_manager" not in routes
    assert await routes["server_info"]({}) == {
        "tp_size": 4,
        "pp_size": 2,
        "dp_size": 1,
        "disaggregation_mode": "null",
    }
    assert engine.loop is asyncio.get_running_loop()
    assert await routes["flush_cache"]({}) == {"timeout_s": None}
    for route in ("init_weights_update_group", "destroy_weights_update_group"):
        assert await routes[route]({"group_name": "trainer"}) == {
            "success": True,
            "message": "trainer",
        }
    assert await routes["async_custom"]({"value": 5}) == {"async": 5}


@pytest.mark.asyncio
async def test_arbitrary_route_is_allowlisted_and_body_cannot_select_method():
    engine = FakeEngine()
    engine.dangerous = Mock()
    routes = _resolved_handlers(engine, ["safe=my_custom_method"])
    replacement = Mock(return_value={"replaced": True})
    engine.my_custom_method = replacement
    body = {"method": "dangerous", "nested": {"ok": True}}

    assert await routes["safe"](body) == {"custom": body}
    replacement.assert_not_called()
    engine.dangerous.assert_not_called()
    assert set(routes) == {"safe"}
    with pytest.raises(ValueError, match="requires a JSON object body"):
        await routes["safe"](["not", "an", "object"])


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_cancelled_sync_engine_route_keeps_engine_routes_serialized():
    owner_loop = asyncio.get_running_loop()
    sync_started = threading.Event()
    release_sync = threading.Event()
    async_engine_started = asyncio.Event()

    class ConcurrentEngine(FakeEngine):
        def blocking_method(self):
            sync_started.set()
            if not release_sync.wait(timeout=2):
                raise TimeoutError("test did not release sync Engine method")
            return {"sync": "done"}

        async def observe_loop(self):
            async_engine_started.set()
            return {"loop_restored": self.loop is owner_loop}

    engine = ConcurrentEngine(owner_loop)
    routes = _resolved_handlers(
        engine,
        [
            "blocking=blocking_method",
            "observe=observe_loop",
            "tm_flush=flush_cache:tm",
        ],
    )
    sync_task = asyncio.create_task(routes["blocking"]({}))
    async with asyncio.timeout(1):
        while not sync_started.is_set():
            await asyncio.sleep(0.001)

    sync_task.cancel()
    async_engine_task = asyncio.create_task(routes["observe"]({}))
    try:
        assert await asyncio.wait_for(routes["tm_flush"]({}), timeout=1) == {
            "timeout_s": None
        }
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(async_engine_started.wait(), timeout=0.1)
    finally:
        release_sync.set()
        sync_outcome, async_engine_outcome = await asyncio.gather(
            sync_task,
            async_engine_task,
            return_exceptions=True,
        )

    assert isinstance(sync_outcome, asyncio.CancelledError)
    assert async_engine_outcome == {"loop_restored": True}
    assert engine.loop is owner_loop


@pytest.mark.asyncio
async def test_typed_tm_request_preserves_weight_version_and_injects_none_request():
    engine = FakeEngine()
    routes = _resolved_handlers(engine, ["update_weights_from_distributed:tm"])
    body = {
        "names": ["model.layers.0.weight"],
        "dtypes": ["float16"],
        "shapes": [[2, 2]],
        "weight_version": "step-42",
    }

    result = await routes["update_weights_from_distributed"](body)

    request_obj, http_request = engine.tokenizer_manager.update_calls[0]
    assert request_obj == UpdateWeightsRequest(
        names=body["names"],
        dtypes=body["dtypes"],
        shapes=body["shapes"],
        weight_version="step-42",
    )
    assert http_request is None
    assert result == {"success": False, "message": "rejected step-42"}
    engine.tokenizer_manager.auto_create_handle_loop.assert_called_once_with()


@pytest.mark.asyncio
async def test_typed_tm_empty_and_populated_requests():
    engine = FakeEngine()
    routes = _resolved_handlers(
        engine,
        [
            "pause_generation:tm",
            "continue_generation:tm",
            "flush=flush_cache:tm",
        ],
    )

    assert await routes["pause_generation"]({}) == {"status": "ok"}
    assert await routes["continue_generation"]({"torch_empty_cache": False}) == {
        "status": "ok"
    }
    assert engine.tokenizer_manager.pause_calls == [
        PauseGenerationRequest(mode="abort")
    ]
    assert engine.tokenizer_manager.continue_calls == [
        ContinueGenerationRequest(torch_empty_cache=False)
    ]
    assert await routes["flush"]({"timeout_s": 3.5}) == {"timeout_s": 3.5}


@pytest.mark.parametrize(
    ("descriptor", "message"),
    [
        ("missing", "has no method 'missing'"),
        ("value", "is not callable"),
        ("missing:tm", "has no method 'missing'"),
    ],
)
def test_startup_rejects_missing_and_non_callable_methods(descriptor, message):
    engine = FakeEngine()
    engine.value = 42

    with pytest.raises(ValueError, match=message):
        resolve_configured_engine_routes(engine, [descriptor])


def test_startup_rejects_missing_tokenizer_manager():
    with pytest.raises(ValueError, match="has no tokenizer_manager"):
        resolve_configured_engine_routes(SimpleNamespace(), ["pause_generation:tm"])


def test_normalize_preserves_failures_and_nested_cuda_graph_config():
    @dataclasses.dataclass
    class PhaseConfig:
        backend: str
        max_bs: int | None
        bs: list[int] | None

    @dataclasses.dataclass
    class CudaGraphConfig:
        decode: PhaseConfig
        prefill: PhaseConfig

    result = normalize_engine_route_result(
        {
            "success": False,
            "typed_failure": TypedFailure(success=False, message="bad"),
            "cuda_graph_config": CudaGraphConfig(
                decode=PhaseConfig("full", 32, [1, 2, 4]),
                prefill=PhaseConfig("breakable", None, [1]),
            ),
            "rank_ids": (0, 1),
            "active_batches": {7, 9},
        }
    )

    assert result["success"] is False
    assert result["typed_failure"] == {"success": False, "message": "bad"}
    assert result["cuda_graph_config"] == {
        "decode": {"backend": "full", "max_bs": 32, "bs": [1, 2, 4]},
        "prefill": {"backend": "breakable", "max_bs": None, "bs": [1]},
    }
    assert result["rank_ids"] == [0, 1]
    assert set(result["active_batches"]) == {7, 9}
    json.dumps(result)
    assert normalize_engine_route_result((False, "failed")) == {
        "success": False,
        "message": "failed",
    }

    @dataclasses.dataclass
    class Recursive:
        nested: object = None

    recursive = Recursive()
    recursive.nested = {"self": recursive}
    result = normalize_engine_route_result(recursive)
    assert result == {"nested": {"self": "<recursive reference>"}}
    json.dumps(result)
