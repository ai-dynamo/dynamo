# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for RL admin request-plane endpoints."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

TRUE_ENV_VALUES = {"1", "true", "yes", "on"}

RLRouteHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]
EndpointGenerator = Callable[[dict[str, Any]], AsyncIterator[dict[str, Any] | None]]


class RLAdminValidationError(ValueError):
    """Validation error whose message can be returned directly to RL clients."""


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable using Dynamo's common true values."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in TRUE_ENV_VALUES


async def first_endpoint_response(
    endpoint_handler: EndpointGenerator,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Return the first response from an async-generator endpoint handler."""
    async for response in endpoint_handler(body):
        return response or {"status": "ok"}
    return {"status": "ok"}


def require_lora_load_request(request: Mapping[str, Any] | None) -> tuple[str, str]:
    """Validate the shared URI-based LoRA load request shape."""
    if request is None or not isinstance(request, Mapping):
        raise RLAdminValidationError(
            "Request is required with 'lora_name' and 'source.uri'"
        )

    lora_name = request.get("lora_name")
    if not lora_name:
        raise RLAdminValidationError("'lora_name' is required in request")

    source = request.get("source")
    if not source or not isinstance(source, Mapping):
        raise RLAdminValidationError("'source' object is required in request")

    lora_uri = source.get("uri")
    if not lora_uri:
        raise RLAdminValidationError("'source.uri' is required in request")

    return str(lora_name), str(lora_uri)


def require_lora_unload_request(request: Mapping[str, Any] | None) -> str:
    """Validate the shared LoRA unload request shape."""
    if request is None or not isinstance(request, Mapping):
        raise RLAdminValidationError("Request is required with 'lora_name' field")

    lora_name = request.get("lora_name")
    if not lora_name:
        raise RLAdminValidationError("'lora_name' is required in request")

    return str(lora_name)


class RLRouteRegistry:
    """Registry for worker RL admin route descriptors."""

    def __init__(
        self,
        runtime: Any,
        *,
        logger_: logging.Logger | None = None,
    ) -> None:
        self._runtime = runtime
        self._logger = logger_ or logger
        self.routes: dict[str, RLRouteHandler] = {}

    def add_route(self, name: str, handler: RLRouteHandler) -> None:
        self.routes[name] = handler

    def add_routes(self, routes: Mapping[str, RLRouteHandler]) -> None:
        for name, handler in routes.items():
            self.add_route(name, handler)

    def describe(self) -> dict[str, Any]:
        response: dict[str, Any] = {
            "status": "ok",
            "routes": sorted(self.routes),
        }

        system_url_fn = getattr(self._runtime, "system_status_server_url", None)
        if callable(system_url_fn):
            system_url = system_url_fn()
            if system_url:
                response["system_url"] = system_url

        return response

    async def dispatch(
        self, request: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        if request is None or not isinstance(request, Mapping):
            return {"status": "error", "message": "rl_dispatch: request required"}

        method = request.get("method")

        if not isinstance(method, str) or not method:
            return {"status": "error", "message": "rl_dispatch: missing 'method' (str)"}

        if method != "routes":
            return {
                "status": "error",
                "method": method,
                "message": "rl request-plane endpoint only supports method='routes'",
            }

        if "kwargs" in request and not isinstance(request.get("kwargs"), Mapping):
            return {
                "status": "error",
                "method": method,
                "message": "rl_dispatch: 'kwargs' must be an object",
            }

        return self.describe()

    async def dispatch_stream(
        self, request: Mapping[str, Any] | None = None
    ) -> AsyncIterator[dict[str, Any]]:
        yield await self.dispatch(request)


def register_rl_routes(
    runtime: Any,
    registry: RLRouteRegistry,
    routes: Mapping[str, RLRouteHandler],
    *,
    enable_dispatch: bool,
) -> None:
    """Register worker system routes and optionally expose route descriptors."""
    for name, handler in routes.items():
        runtime.register_engine_route(name, handler)
        if enable_dispatch:
            registry.add_route(name, handler)
