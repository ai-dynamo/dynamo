# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for load_in_process_plugins."""

from __future__ import annotations

import pytest

from dynamo.planner.plugins.orchestrator.in_process_loader import (
    load_in_process_plugins,
)
from dynamo.planner.plugins.registry.config import InProcessPluginSpec

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


FAKE_PLUGIN_MODULE = "dynamo.planner.tests.plugins.orchestrator._fake_in_process_plugin"


def test_loader_registers_plugin_from_module_path(ctx_factory):
    ctx = ctx_factory()
    spec = InProcessPluginSpec.model_validate(
        {
            "module": FAKE_PLUGIN_MODULE,
            "class": "FakePlugin",
            "plugin_id": "fake1",
            "plugin_type": "propose",
            "priority": 10,
            "execution_interval_seconds": 5.0,
            "hold_policy": "HOLD_LAST",
            "kwargs": {"tag": "alpha"},
        }
    )
    load_in_process_plugins(ctx["orchestrator"], [spec])
    (info,) = ctx["orchestrator"].list_plugins()
    assert info.plugin_id == "fake1"
    assert info.transport == "in_process"
    # is_builtin=False since loader sets it that way for user plugins.
    assert info.is_builtin is False


def test_loader_propagates_kwargs_to_instance(ctx_factory):
    ctx = ctx_factory()
    spec = InProcessPluginSpec.model_validate(
        {
            "module": FAKE_PLUGIN_MODULE,
            "class": "FakePlugin",
            "plugin_id": "fake_kw",
            "plugin_type": "propose",
            "priority": 10,
            "kwargs": {"tag": "configured-tag"},
        }
    )
    load_in_process_plugins(ctx["orchestrator"], [spec])
    plugin = ctx["registry"].get_plugin("fake_kw")
    assert plugin is not None
    # The instance is wrapped inside InProcessTransport; retrieve the
    # underlying object to verify kwargs landed.
    instance = plugin.transport._instance  # noqa: SLF001 — test-only
    assert instance.tag == "configured-tag"


def test_loader_raises_on_unknown_module(ctx_factory):
    ctx = ctx_factory()
    spec = InProcessPluginSpec.model_validate(
        {
            "module": "dynamo.planner.tests.plugins.orchestrator.does_not_exist",
            "class": "Missing",
            "plugin_id": "x",
            "plugin_type": "propose",
            "priority": 1,
        }
    )
    with pytest.raises(ImportError, match="failed to import module"):
        load_in_process_plugins(ctx["orchestrator"], [spec])


def test_loader_raises_on_unknown_class(ctx_factory):
    ctx = ctx_factory()
    spec = InProcessPluginSpec.model_validate(
        {
            "module": FAKE_PLUGIN_MODULE,
            "class": "NoSuchClass",
            "plugin_id": "y",
            "plugin_type": "propose",
            "priority": 1,
        }
    )
    with pytest.raises(AttributeError, match="no attribute"):
        load_in_process_plugins(ctx["orchestrator"], [spec])


def test_loader_wraps_construction_failure_with_context(ctx_factory):
    """``cls(**kwargs)`` failure (e.g. typo in kwarg name) must surface a
    RuntimeError naming the plugin_id + class + kwargs, so operators can
    identify the offending YAML entry without parsing a raw TypeError
    traceback."""
    ctx = ctx_factory()
    spec = InProcessPluginSpec.model_validate(
        {
            "module": FAKE_PLUGIN_MODULE,
            "class": "FakePlugin",
            "plugin_id": "bad-kwargs",
            "plugin_type": "propose",
            "priority": 1,
            # FakePlugin.__init__(self, tag: str = "default") — `unknown_kw`
            # is not a valid parameter.
            "kwargs": {"unknown_kw": "value"},
        }
    )
    with pytest.raises(RuntimeError) as exc_info:
        load_in_process_plugins(ctx["orchestrator"], [spec])
    # Error message must include enough breadcrumbs to identify the entry.
    msg = str(exc_info.value)
    assert "bad-kwargs" in msg
    assert "FakePlugin" in msg
    assert "unknown_kw" in msg
    # Underlying cause must be preserved.
    assert isinstance(exc_info.value.__cause__, TypeError)


def test_loader_loads_multiple_specs(ctx_factory):
    ctx = ctx_factory()
    specs = [
        InProcessPluginSpec.model_validate(
            {
                "module": FAKE_PLUGIN_MODULE,
                "class": "FakePlugin",
                "plugin_id": f"fake{i}",
                "plugin_type": "propose",
                "priority": i,
            }
        )
        for i in range(3)
    ]
    load_in_process_plugins(ctx["orchestrator"], specs)
    ids = sorted(i.plugin_id for i in ctx["orchestrator"].list_plugins())
    assert ids == ["fake0", "fake1", "fake2"]


def test_loader_passes_scale_interval_fields_to_registered_plugin(ctx_factory):
    """``InProcessPluginSpec`` now exposes ``needs`` /
    ``requires_produced_fields`` / ``observation_window_seconds`` so
    ConfigMap-driven in-process plugins can declare the scale_interval
    cadence contract.  Without the loader-side passthrough a
    ``throughput_propose`` asking for ``requires_produced_fields=
    ["predictions"]`` would have fired every tick regardless of
    upstream predict output.
    """
    ctx = ctx_factory()
    spec = InProcessPluginSpec.model_validate(
        {
            "module": FAKE_PLUGIN_MODULE,
            "class": "FakePlugin",
            "plugin_id": "throughput_propose",
            "plugin_type": "propose",
            "priority": 100,
            "execution_interval_seconds": 60.0,
            "needs": ["observations.traffic"],
            "requires_produced_fields": ["predictions"],
            "observation_window_seconds": 180.0,
        }
    )
    load_in_process_plugins(ctx["orchestrator"], [spec])
    plugin = ctx["orchestrator"].registry.get_plugin("throughput_propose")
    assert plugin is not None
    assert plugin.needs == ["observations.traffic"]
    assert plugin.requires_produced_fields == ["predictions"]
    assert plugin.observation_window_seconds == 180.0
    assert plugin.execution_interval_seconds == 60.0
