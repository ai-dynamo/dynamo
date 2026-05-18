# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from dynamo.common.rl import (
    RLAdminValidationError,
    RLRouteRegistry,
    first_endpoint_response,
    register_rl_routes,
    require_lora_load_request,
    require_lora_unload_request,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class _Runtime:
    def __init__(self, system_url: str | None = None) -> None:
        self.system_url = system_url
        self.registered: list[tuple[str, object]] = []

    def system_status_server_url(self) -> str | None:
        return self.system_url

    def register_engine_route(self, name: str, handler: object) -> None:
        self.registered.append((name, handler))


def test_route_registry_describes_routes() -> None:
    runtime = _Runtime("http://worker:8081")
    registry = RLRouteRegistry(runtime)

    async def ping(body: dict) -> dict:
        return {"status": "ok", "body": body}

    registry.add_route("ping", ping)

    routes = asyncio.run(registry.dispatch({"method": "routes"}))
    assert routes == {
        "status": "ok",
        "routes": ["ping"],
        "system_url": "http://worker:8081",
    }

    routes_with_kwargs = asyncio.run(
        registry.dispatch({"method": "routes", "kwargs": {}})
    )
    assert routes_with_kwargs == routes


def test_route_registry_rejects_request_plane_admin_execution() -> None:
    registry = RLRouteRegistry(_Runtime())

    response = asyncio.run(registry.dispatch({"method": "missing"}))

    assert response["status"] == "error"
    assert response["method"] == "missing"
    assert (
        response["message"] == "rl request-plane endpoint only supports method='routes'"
    )


def test_route_registry_rejects_non_object_kwargs_for_routes() -> None:
    registry = RLRouteRegistry(_Runtime())

    response = asyncio.run(registry.dispatch({"method": "routes", "kwargs": []}))

    assert response == {
        "status": "error",
        "method": "routes",
        "message": "rl_dispatch: 'kwargs' must be an object",
    }


def test_register_rl_routes_always_registers_engine_route() -> None:
    runtime = _Runtime()
    registry = RLRouteRegistry(runtime)

    async def ping(body: dict) -> dict:
        return {"status": "ok", "body": body}

    register_rl_routes(runtime, registry, {"ping": ping}, enable_dispatch=False)

    assert runtime.registered == [("ping", ping)]
    assert registry.routes == {}

    register_rl_routes(runtime, registry, {"ping": ping}, enable_dispatch=True)

    assert registry.routes == {"ping": ping}


def test_first_endpoint_response_returns_first_chunk() -> None:
    async def endpoint(_body: dict):
        yield {"status": "ok", "value": 1}
        yield {"status": "ok", "value": 2}

    response = asyncio.run(first_endpoint_response(endpoint, {}))

    assert response == {"status": "ok", "value": 1}


def test_lora_load_request_validation() -> None:
    assert require_lora_load_request(
        {"lora_name": "adapter", "source": {"uri": "file:///tmp/adapter"}}
    ) == ("adapter", "file:///tmp/adapter")

    try:
        require_lora_load_request({"lora_name": "adapter"})
    except RLAdminValidationError as exc:
        assert str(exc) == "'source' object is required in request"
    else:
        raise AssertionError("expected validation error")


def test_lora_unload_request_validation() -> None:
    assert require_lora_unload_request({"lora_name": "adapter"}) == "adapter"

    try:
        require_lora_unload_request({})
    except RLAdminValidationError as exc:
        assert str(exc) == "'lora_name' is required in request"
    else:
        raise AssertionError("expected validation error")
