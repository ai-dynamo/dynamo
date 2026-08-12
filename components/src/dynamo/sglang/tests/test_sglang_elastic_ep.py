# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SGLang worker's elastic-EP scale-up control.

The validation paths are exercised without a real SGLang engine (they return
before the ``ScaleElasticEPReqInput`` import), and the success path runs
against a stubbed ``sglang.srt.managers.io_struct`` module so it does not
require the elastic-EP-capable SGLang 0.5.16+.
"""

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture(autouse=True)
def _stub_sglang_io_struct(monkeypatch):
    """Provide ScaleElasticEPReqInput without the CUDA-only sglang import.

    The handler imports ``ScaleElasticEPReqInput`` lazily (it only exists on
    SGLang >= 0.5.16), so stubbing it lets the success-path test run without a
    real sglang install and regardless of the installed version.
    """

    io_struct = types.ModuleType("sglang.srt.managers.io_struct")

    class ScaleElasticEPReqInput:
        def __init__(self, new_ep_size):
            self.new_ep_size = new_ep_size

    io_struct.ScaleElasticEPReqInput = ScaleElasticEPReqInput
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_struct)


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


def _make_handler(
    *,
    backend="mooncake",
    supports=True,
    scale_result=None,
    scale_side_effect=None,
    state=None,
):
    """Build a handler bound to a fake SGLang engine, bypassing __init__."""

    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler._scale_ep_lock = asyncio.Lock()

    tokenizer_manager = SimpleNamespace(
        server_args=SimpleNamespace(elastic_ep_backend=backend),
        get_elastic_ep_state=MagicMock(
            return_value=state
            if state is not None
            else {
                "is_scaling_elastic_ep": False,
                "effective_ep_size": 6,
                "scale_phase": "serving_expanded",
                "last_error": None,
            }
        ),
    )
    if supports:
        if scale_result is None:
            scale_result = SimpleNamespace(
                success=True,
                message="Scale up completed, new EP size: 6",
                old_ep_size=4,
                new_ep_size=6,
                pending_ep_size=None,
            )
        tokenizer_manager.scale_elastic_ep = AsyncMock(
            return_value=scale_result, side_effect=scale_side_effect
        )

    handler.engine = SimpleNamespace(tokenizer_manager=tokenizer_manager)
    # register_engine_routes() wires the model-taint route from generate_endpoint;
    # None is fine here since the route's handler is never invoked in these tests.
    handler.generate_endpoint = None
    return handler


@pytest.mark.asyncio
async def test_scale_elastic_ep_rejects_missing_size():
    handler = _make_handler()

    result = await handler.scale_elastic_ep({})

    assert result == {
        "status": "error",
        "message": "Missing required field: new_ep_size",
    }
    handler.engine.tokenizer_manager.scale_elastic_ep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", [True, False, "6", 6.0, None])
async def test_scale_elastic_ep_rejects_non_integer(bad_value):
    handler = _make_handler()

    result = await handler.scale_elastic_ep({"new_ep_size": bad_value})

    assert result["status"] == "error"
    handler.engine.tokenizer_manager.scale_elastic_ep.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", [0, -2])
async def test_scale_elastic_ep_rejects_non_positive(bad_value):
    handler = _make_handler()

    result = await handler.scale_elastic_ep({"new_ep_size": bad_value})

    assert result == {
        "status": "error",
        "message": "new_ep_size must be a positive integer",
    }
    handler.engine.tokenizer_manager.scale_elastic_ep.assert_not_awaited()


@pytest.mark.asyncio
async def test_scale_elastic_ep_requires_backend():
    handler = _make_handler(backend=None)

    result = await handler.scale_elastic_ep({"new_ep_size": 6})

    assert result == {
        "status": "error",
        "message": "elastic EP is not enabled (set --elastic-ep-backend)",
    }
    handler.engine.tokenizer_manager.scale_elastic_ep.assert_not_awaited()


@pytest.mark.asyncio
async def test_scale_elastic_ep_success_forwards_new_ep_size():
    handler = _make_handler()

    result = await handler.scale_elastic_ep({"new_ep_size": 6})

    assert result == {
        "status": "ok",
        "message": "Scale up completed, new EP size: 6",
        "old_ep_size": 4,
        "new_ep_size": 6,
    }
    handler.engine.tokenizer_manager.scale_elastic_ep.assert_awaited_once()
    (req,) = handler.engine.tokenizer_manager.scale_elastic_ep.await_args.args
    assert req.new_ep_size == 6


@pytest.mark.asyncio
async def test_scale_elastic_ep_reports_engine_rejection():
    scale_result = SimpleNamespace(
        success=False,
        message="joining group not ready",
        old_ep_size=4,
        new_ep_size=None,
        pending_ep_size=5,
    )
    handler = _make_handler(scale_result=scale_result)

    result = await handler.scale_elastic_ep({"new_ep_size": 6})

    assert result == {
        "status": "error",
        "message": "joining group not ready",
        "old_ep_size": 4,
        "new_ep_size": None,
        "pending_ep_size": 5,
    }


@pytest.mark.asyncio
async def test_scale_elastic_ep_reports_exception():
    handler = _make_handler(scale_side_effect=RuntimeError("mooncake transfer failed"))

    result = await handler.scale_elastic_ep({"new_ep_size": 6})

    assert result == {
        "status": "error",
        "message": "mooncake transfer failed",
    }


@pytest.mark.asyncio
async def test_is_scaling_elastic_ep_returns_state():
    handler = _make_handler()

    result = await handler.is_scaling_elastic_ep({})

    assert result == {
        "is_scaling_elastic_ep": False,
        "effective_ep_size": 6,
        "scale_phase": "serving_expanded",
        "last_error": None,
    }


@pytest.mark.asyncio
async def test_is_scaling_elastic_ep_requires_backend():
    handler = _make_handler(backend=None)

    result = await handler.is_scaling_elastic_ep({})

    assert result == {
        "status": "error",
        "message": "elastic EP is not enabled (set --elastic-ep-backend)",
    }


def test_supports_elastic_ep_gating():
    assert _make_handler(supports=True)._supports_elastic_ep() is True
    assert _make_handler(supports=False)._supports_elastic_ep() is False

    handler = _make_handler()
    handler.engine = None
    assert handler._supports_elastic_ep() is False


def _register_routes(handler):
    registered = {}

    class _Runtime:
        def register_engine_route(self, path, route_handler):
            registered[path] = route_handler

    handler.config = SimpleNamespace(dynamo_args=SimpleNamespace(engine_routes=[]))
    handler.register_engine_routes(_Runtime())
    return registered


def test_register_engine_routes_exposes_elastic_ep_when_supported():
    registered = _register_routes(_make_handler(supports=True))

    assert registered["control/scale_elastic_ep"] is not None
    assert registered["control/is_scaling_elastic_ep"] is not None


def test_register_engine_routes_hides_elastic_ep_when_unsupported():
    registered = _register_routes(_make_handler(supports=False))

    assert "control/scale_elastic_ep" not in registered
    assert "control/is_scaling_elastic_ep" not in registered
    # Sanity: the always-present controls are still registered.
    assert "control/start_profile" in registered
