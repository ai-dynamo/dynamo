# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``NativePlannerBase._apply_scaling_targets`` accounting.

The method is the single funnel both engine paths (PSM + orchestrator)
push targets through, so its label-classification is the ground truth
operators consult on dashboards. The test focuses on the
``ConnectorBusyError`` skip path that A1 introduced — pre-A1 the
connector silent-returned, the funnel never saw an exception, and
``execute_total{result=success}`` was incremented while no scaling
actually happened.
"""

from __future__ import annotations

import types
from typing import Any

import pytest

from dynamo.planner.core.base import NativePlannerBase
from dynamo.planner.errors import ConnectorBusyError

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


class _LabelCounter:
    """Minimal stand-in for ``Counter.labels(...).inc()`` chains."""

    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def labels(self, **kwargs: Any) -> "_LabelCounter":
        self.calls.append(dict(kwargs))
        return self

    def inc(self) -> None:
        pass

    def observe(self, value: float) -> None:
        pass


class _StubMetrics:
    """Just the three metric handles ``_apply_scaling_targets`` touches."""

    def __init__(self) -> None:
        self.execute_total = _LabelCounter()
        self.execute_latency_seconds = _LabelCounter()
        self.execute_skip_reason_total = _LabelCounter()


class _StubConfig:
    advisory = False


class _StubConnector:
    """Lets a test inject any side-effect into ``set_component_replicas``."""

    def __init__(self, side_effect: Any = None) -> None:
        self._side_effect = side_effect
        self.calls: int = 0

    async def set_component_replicas(self, targets, blocking: bool = False) -> None:
        self.calls += 1
        if isinstance(self._side_effect, Exception):
            raise self._side_effect
        if callable(self._side_effect):
            await self._side_effect(targets, blocking=blocking)


def _make_self(connector: _StubConnector, advisory: bool = False) -> Any:
    cfg = _StubConfig()
    cfg.advisory = advisory  # type: ignore[attr-defined]
    s = types.SimpleNamespace(
        prometheus_metrics=_StubMetrics(),
        config=cfg,
        connector=connector,
    )
    return s


def _result_labels(stub: Any) -> list[str]:
    return [c.get("result") for c in stub.prometheus_metrics.execute_total.calls]


def _skip_reasons(stub: Any) -> list[str]:
    return [
        c.get("reason")
        for c in stub.prometheus_metrics.execute_skip_reason_total.calls
    ]


@pytest.mark.asyncio
async def test_connector_busy_records_skip_label_not_success():
    """Core regression: ``ConnectorBusyError`` must hit
    ``result=skipped_connector_blocked``, NOT ``result=success`` or
    ``result=error``."""
    connector = _StubConnector(
        side_effect=ConnectorBusyError(
            reason="deployment_not_ready", detail="DGD foo not ready"
        )
    )
    stub = _make_self(connector)

    await NativePlannerBase._apply_scaling_targets(stub, [object()])

    assert connector.calls == 1
    assert _result_labels(stub) == ["skipped_connector_blocked"]
    assert _skip_reasons(stub) == ["deployment_not_ready"]


@pytest.mark.asyncio
async def test_connector_busy_does_not_re_raise():
    """``ConnectorBusyError`` is a transient signal, not a fatal error;
    ``_apply_scaling_targets`` swallows it so the next tick can retry."""
    connector = _StubConnector(
        side_effect=ConnectorBusyError(reason="deployment_not_ready")
    )
    stub = _make_self(connector)

    await NativePlannerBase._apply_scaling_targets(stub, [object()])  # no raise


@pytest.mark.asyncio
async def test_unrelated_exception_records_error_and_re_raises():
    """Generic exceptions still go to ``result=error`` and propagate."""
    boom = RuntimeError("apiserver down")
    connector = _StubConnector(side_effect=boom)
    stub = _make_self(connector)

    with pytest.raises(RuntimeError):
        await NativePlannerBase._apply_scaling_targets(stub, [object()])

    assert _result_labels(stub) == ["error"]


@pytest.mark.asyncio
async def test_success_path_unaffected_by_a1_change():
    """Sanity-check: when the connector returns normally, we still
    record ``result=success``. A1 must not regress the happy path."""
    connector = _StubConnector(side_effect=None)
    stub = _make_self(connector)

    await NativePlannerBase._apply_scaling_targets(stub, [object()])

    assert _result_labels(stub) == ["success"]


@pytest.mark.asyncio
async def test_busy_reason_is_used_as_skip_reason_label():
    """``ConnectorBusyError.reason`` becomes the
    ``execute_skip_reason_total{reason}`` label verbatim — that is the
    contract, so non-K8s connectors can plug in their own bounded
    reason strings (e.g. ``virtual_connector_paused``) without
    touching this funnel."""
    connector = _StubConnector(
        side_effect=ConnectorBusyError(reason="virtual_connector_paused")
    )
    stub = _make_self(connector)

    await NativePlannerBase._apply_scaling_targets(stub, [object()])

    assert _skip_reasons(stub) == ["virtual_connector_paused"]
