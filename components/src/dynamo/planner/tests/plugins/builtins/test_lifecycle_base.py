# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for PluginLifecycle ABC + BuiltinPluginBase (PR 6 6-1)."""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.builtins.base import BuiltinPluginBase
from dynamo.planner.plugins.lifecycle import PluginLifecycle
from dynamo.planner.plugins.types import (
    BootstrapRequest,
    BootstrapResponse,
    ResetRequest,
    ResetResponse,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _FakeOrchestrator:
    """Minimal orchestrator stub exposing only the regression surface
    BuiltinPluginBase forwards to."""

    def __init__(self):
        self._store: dict = {}

    def get_regression(self, kind):
        return self._store.get(kind)

    def update_regression(self, kind, model):
        self._store[kind] = model


class _FakeConfig:
    pass


# ---------------------------------------------------------------------------
# Abstract surface
# ---------------------------------------------------------------------------


def test_plugin_lifecycle_abstract_cannot_instantiate():
    with pytest.raises(TypeError):
        PluginLifecycle()  # type: ignore[abstract]


class _MissingReset(PluginLifecycle):
    async def Bootstrap(self, request):
        return BootstrapResponse(ok=True)


def test_subclass_must_implement_both_bootstrap_and_reset():
    with pytest.raises(TypeError):
        _MissingReset()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# BuiltinPluginBase
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_default_bootstrap_and_reset_return_ok():
    orch = _FakeOrchestrator()
    plugin = BuiltinPluginBase(orch, _FakeConfig())  # type: ignore[arg-type]
    b_resp = await plugin.Bootstrap(BootstrapRequest())
    assert isinstance(b_resp, BootstrapResponse)
    assert b_resp.ok is True
    r_resp = await plugin.Reset(ResetRequest())
    assert isinstance(r_resp, ResetResponse)
    assert r_resp.ok is True


def test_get_regression_forwards_to_orchestrator():
    orch = _FakeOrchestrator()
    orch._store["prefill"] = "model-p"
    plugin = BuiltinPluginBase(orch, _FakeConfig())  # type: ignore[arg-type]
    assert plugin.get_regression("prefill") == "model-p"
    assert plugin.get_regression("absent") is None


def test_update_regression_writes_through_orchestrator():
    orch = _FakeOrchestrator()
    plugin = BuiltinPluginBase(orch, _FakeConfig())  # type: ignore[arg-type]
    plugin.update_regression("decode", "model-d")
    assert orch.get_regression("decode") == "model-d"


# ---------------------------------------------------------------------------
# Subclass can override lifecycle with real state
# ---------------------------------------------------------------------------


class _StatefulPlugin(BuiltinPluginBase):
    def __init__(self, orch, config):
        super().__init__(orch, config)
        self.counter = 0

    async def Bootstrap(self, request):
        self.counter = 42
        return BootstrapResponse(ok=True, message="seeded")

    async def Reset(self, request):
        self.counter = 0
        return ResetResponse(ok=True, message="zeroed")


@pytest.mark.asyncio
async def test_subclass_bootstrap_overrides_default():
    plugin = _StatefulPlugin(_FakeOrchestrator(), _FakeConfig())  # type: ignore[arg-type]
    assert plugin.counter == 0
    resp = await plugin.Bootstrap(BootstrapRequest())
    assert plugin.counter == 42
    assert resp.message == "seeded"


@pytest.mark.asyncio
async def test_subclass_reset_overrides_default():
    plugin = _StatefulPlugin(_FakeOrchestrator(), _FakeConfig())  # type: ignore[arg-type]
    plugin.counter = 99
    resp = await plugin.Reset(ResetRequest())
    assert plugin.counter == 0
    assert resp.message == "zeroed"
