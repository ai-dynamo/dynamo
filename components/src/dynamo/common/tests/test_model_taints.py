# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from dynamo.common import model_taints

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


class _Runtime:
    def __init__(self) -> None:
        self.route_name: str | None = None
        self.handler = None

    def register_engine_route(self, name, handler) -> None:
        self.route_name = name
        self.handler = handler


def test_model_taint_route_updates_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    update = AsyncMock()
    monkeypatch.setattr(model_taints, "update_model_taints", update)
    runtime = _Runtime()
    endpoint = object()

    model_taints.register_model_taint_route(runtime, endpoint)

    assert runtime.route_name == "update/model_taints"
    response = asyncio.run(
        runtime.handler({"taints": ["capacity/fast", "capacity/fast"]})
    )
    assert response == {
        "status": "ok",
        "taints": ["capacity/fast"],
    }
    update.assert_awaited_once_with(endpoint, {"capacity/fast"})


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({}, "'taints' must be a JSON array of strings"),
        ({"taints": "fast"}, "'taints' must be a JSON array of strings"),
        ({"taints": [1]}, "'taints' must be a JSON array of strings"),
        (
            {"taints": ["dynamo.topology/zone=west"]},
            "uses reserved prefix 'dynamo.topology/'",
        ),
    ],
)
def test_model_taint_route_rejects_invalid_requests(body, message) -> None:
    runtime = _Runtime()
    model_taints.register_model_taint_route(runtime, object())

    with pytest.raises(ValueError, match=message):
        asyncio.run(runtime.handler(body))
