# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import types

import pytest

from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _DummyRuntime:
    def __init__(self):
        self.routes = {}

    def register_engine_route(self, name, handler):
        self.routes[name] = handler


class _DummyWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        if False:
            yield request, context


def test_register_engine_routes_includes_weight_update_routes():
    handler = _DummyWorkerHandler.__new__(_DummyWorkerHandler)
    runtime = _DummyRuntime()

    handler.register_engine_routes(runtime)

    assert "init_weights_update_group" in runtime.routes
    assert "destroy_weights_update_group" in runtime.routes
    assert "get_weight_version" in runtime.routes


@pytest.mark.asyncio
async def test_get_weight_version_reads_active_version_from_server_args():
    handler = _DummyWorkerHandler.__new__(_DummyWorkerHandler)
    handler.engine = types.SimpleNamespace(
        tokenizer_manager=types.SimpleNamespace(
            server_args=types.SimpleNamespace(weight_version=17)
        )
    )

    result = await handler.get_weight_version({})

    assert result == {"weight_version": 17}


@pytest.mark.asyncio
async def test_weight_update_routes_return_unsupported_without_engine():
    handler = _DummyWorkerHandler.__new__(_DummyWorkerHandler)
    handler.engine = None

    init_result = await handler.init_weights_update_group({})
    destroy_result = await handler.destroy_weights_update_group({})
    version_result = await handler.get_weight_version({})

    expected = {
        "success": False,
        "message": "weight update control not supported on this worker",
    }
    assert init_result == expected
    assert destroy_result == expected
    assert version_result == expected
