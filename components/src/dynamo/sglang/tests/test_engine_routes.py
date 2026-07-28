# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for configuration-driven SGLang engine routes."""

import argparse
import asyncio
import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import Mock

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


class UpdateWeightsRequest:
    """Msgspec-shaped request containing the vanilla SGLang fields."""

    __struct_fields__ = (
        "names",
        "dtypes",
        "shapes",
        "group_name",
        "weight_version",
    )

    def __init__(
        self,
        *,
        names,
        dtypes,
        shapes,
        group_name="weight_update_group",
        weight_version=None,
    ):
        self.names = names
        self.dtypes = dtypes
        self.shapes = shapes
        self.group_name = group_name
        self.weight_version = weight_version


class TypedFailure:
    __struct_fields__ = ("success", "message")

    def __init__(self, *, success, message):
        self.success = success
        self.message = message


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
        self.custom_calls = []

    def my_custom_method(self, **kwargs):
        self.custom_calls.append(kwargs)
        return {"custom": kwargs}

    async def async_custom_method(self, value):
        return {"async": value}

    async def _server_info(self):
        return {"version": "test"}

    def get_server_info(self):
        return self.loop.run_until_complete(self._server_info())

    def flush_cache(self):
        return self.loop.run_until_complete(self.tokenizer_manager.flush_cache())

    def init_weights_update_group(self, **kwargs):
        return True, kwargs["group_name"]

    def destroy_weights_update_group(self, **kwargs):
        return True, kwargs["group_name"]


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

    parser = argparse.ArgumentParser()
    DynamoSGLangArgGroup().add_arguments(parser)
    args = parser.parse_args(
        [
            "--engine-route",
            "server_info=get_server_info",
            "--engine-route",
            "pause_generation:tm",
        ]
    )
    assert args.engine_routes == [
        "server_info=get_server_info",
        "pause_generation:tm",
    ]

    monkeypatch.setenv(
        "DYN_SGLANG_ENGINE_ROUTES",
        "flush_cache update_weights_from_distributed:tm",
    )
    env_parser = argparse.ArgumentParser()
    DynamoSGLangArgGroup().add_arguments(env_parser)
    env_args = env_parser.parse_args([])
    assert env_args.engine_routes == [
        "flush_cache",
        "update_weights_from_distributed:tm",
    ]


def test_compatibility_descriptors_resolve_without_a_method_registry():
    engine = FakeEngine()
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
    }
    assert "call_tokenizer_manager" not in routes


@pytest.mark.asyncio
async def test_compatibility_engine_methods_dispatch():
    engine = FakeEngine(asyncio.get_running_loop())
    routes = _resolved_handlers(
        engine,
        [
            "server_info=get_server_info",
            "flush_cache",
            "init_weights_update_group",
            "destroy_weights_update_group",
        ],
    )

    assert await routes["server_info"]({}) == {"version": "test"}
    assert await routes["flush_cache"]({}) == {"timeout_s": None}
    assert await routes["init_weights_update_group"]({"group_name": "trainer"}) == {
        "success": True,
        "message": "trainer",
    }
    assert await routes["destroy_weights_update_group"]({"group_name": "trainer"}) == {
        "success": True,
        "message": "trainer",
    }


@pytest.mark.asyncio
async def test_arbitrary_custom_engine_method_requires_configuration_only():
    engine = FakeEngine()
    routes = _resolved_handlers(engine, ["custom=my_custom_method"])

    result = await routes["custom"]({"value": 7, "nested": {"ok": True}})

    assert result == {"custom": {"value": 7, "nested": {"ok": True}}}
    assert engine.custom_calls == [{"value": 7, "nested": {"ok": True}}]
    assert "my_custom_method" not in routes


@pytest.mark.asyncio
async def test_request_body_cannot_select_a_different_method():
    engine = FakeEngine()
    engine.dangerous = Mock()
    routes = _resolved_handlers(engine, ["safe=my_custom_method"])

    result = await routes["safe"]({"method": "dangerous"})

    assert result == {"custom": {"method": "dangerous"}}
    engine.dangerous.assert_not_called()
    assert "dangerous" not in routes


@pytest.mark.asyncio
async def test_sync_engine_wrapper_bridges_to_the_running_owner_loop():
    engine = FakeEngine(asyncio.get_running_loop())
    routes = _resolved_handlers(engine, ["server_info=get_server_info"])

    assert await routes["server_info"]({}) == {"version": "test"}
    assert engine.loop is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_async_engine_method_is_awaited():
    routes = _resolved_handlers(FakeEngine(), ["custom=async_custom_method"])

    assert await routes["custom"]({"value": 5}) == {"async": 5}


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
    assert isinstance(request_obj, UpdateWeightsRequest)
    assert request_obj.names == body["names"]
    assert request_obj.dtypes == body["dtypes"]
    assert request_obj.shapes == body["shapes"]
    assert request_obj.weight_version == "step-42"
    assert http_request is None
    assert result == {"success": False, "message": "rejected step-42"}
    engine.tokenizer_manager.auto_create_handle_loop.assert_called_once_with()


@pytest.mark.asyncio
async def test_typed_tm_empty_and_populated_requests():
    engine = FakeEngine()
    routes = _resolved_handlers(
        engine, ["pause_generation:tm", "continue_generation:tm"]
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


@pytest.mark.asyncio
async def test_untyped_tm_method_receives_body_as_kwargs():
    engine = FakeEngine()
    routes = _resolved_handlers(engine, ["flush=flush_cache:tm"])

    assert await routes["flush"]({"timeout_s": 3.5}) == {"timeout_s": 3.5}


@pytest.mark.asyncio
async def test_non_object_body_is_rejected():
    route = _resolved_handlers(FakeEngine(), ["custom=my_custom_method"])["custom"]

    with pytest.raises(ValueError, match="requires a JSON object body"):
        await route(["not", "an", "object"])


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


def test_startup_resolves_methods_once():
    engine = FakeEngine()
    routes = _resolved_handlers(engine, ["custom=my_custom_method"])
    engine.my_custom_method = Mock(return_value={"replaced": True})

    assert routes["custom"]._method.__self__ is engine
    assert routes["custom"]._method.__func__ is FakeEngine.my_custom_method


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


def test_normalize_tuple_failure_and_recursive_value():
    @dataclasses.dataclass
    class Recursive:
        nested: object = None

    recursive = Recursive()
    recursive.nested = {"self": recursive}

    assert normalize_engine_route_result((False, "failed")) == {
        "success": False,
        "message": "failed",
    }
    result = normalize_engine_route_result(recursive)
    assert result == {"nested": {"self": "<recursive reference>"}}
    json.dumps(result)
