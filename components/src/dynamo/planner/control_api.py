# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local runtime API for planner minimum endpoint configuration."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from aiohttp import web
from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


class MinimumEndpointValidationError(ValueError):
    """Raised when a well-formed runtime update is invalid for the planner."""


class MinimumEndpointController(Protocol):
    async def get_min_endpoints(self) -> dict[str, Any]:
        ...

    async def patch_min_endpoints(self, updates: dict[str, int]) -> dict[str, Any]:
        ...


class _MinimumEndpointsPatch(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    min_endpoint: int | None = Field(default=None, ge=1)
    prefill_min_endpoint: int | None = Field(default=None, ge=1)
    decode_min_endpoint: int | None = Field(default=None, ge=1)


def _error(message: str, status: int) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _build_app(controller: MinimumEndpointController) -> web.Application:
    async def handle_get(_request: web.Request) -> web.Response:
        return web.json_response(await controller.get_min_endpoints())

    async def handle_patch(request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except Exception:
            return _error("request body must be valid JSON", 400)
        if not isinstance(payload, dict):
            return _error("request body must be a JSON object", 400)

        try:
            patch = _MinimumEndpointsPatch.model_validate(payload)
        except ValidationError as exc:
            # Unknown fields make the patch structurally invalid (400). Type,
            # range, and null errors on known fields are semantic validation
            # failures (422).
            status = (
                400
                if any(error["type"] == "extra_forbidden" for error in exc.errors())
                else 422
            )
            return _error(str(exc), status)

        fields = patch.model_fields_set
        if not fields:
            return _error("request body must include at least one endpoint field", 400)
        if any(getattr(patch, field) is None for field in fields):
            return _error("minimum endpoint fields cannot be null", 422)

        updates = {field: int(getattr(patch, field)) for field in fields}
        try:
            response = await controller.patch_min_endpoints(updates)
        except MinimumEndpointValidationError as exc:
            return _error(str(exc), 422)
        return web.json_response(response)

    app = web.Application()
    app.router.add_get("/v1/min-endpoints", handle_get)
    app.router.add_patch("/v1/min-endpoints", handle_patch)
    return app


async def start_control_api(
    controller: MinimumEndpointController, port: int
) -> web.AppRunner:
    """Start the localhost-only runtime configuration API."""

    runner = web.AppRunner(_build_app(controller))
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    try:
        await site.start()
    except Exception:
        await runner.cleanup()
        raise
    logger.info(
        "Planner runtime configuration API listening on "
        "http://127.0.0.1:%s/v1/min-endpoints",
        port,
    )
    return runner
