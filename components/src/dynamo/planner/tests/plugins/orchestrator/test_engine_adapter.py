# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for ``OrchestratorEngineAdapter`` cadence parity.

Covers two PSM-parity bugs caught after K8s smoke v14:

- ``initial_tick`` previously read ``self._config.throughput_adjustment_interval``
  (missing ``_seconds`` suffix). The Pydantic ``validation_alias`` only affects
  input parsing — attribute access requires the canonical name. Triggered an
  ``AttributeError`` whenever ``enable_throughput_scaling=True``.

- ``_MERGE_TOLERANCE_S`` was set to ``1e-9`` (float epsilon framing) instead
  of PSM's ``0.5`` (wall-clock-drift padding). With the tight tolerance a
  load tick and a throughput tick scheduled within ~ms of each other failed
  to merge — splitting into 2 ticks where PSM produces 1.
"""

from __future__ import annotations

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    EngineCapabilities,
    ScheduledTick,
    WorkerCapabilities,
)
from dynamo.planner.plugins.orchestrator.engine_adapter import (
    OrchestratorEngineAdapter,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def _caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384
        )
    )


def _agg_config_throughput_on() -> PlannerConfig:
    # SLA mode keeps ``enable_throughput_scaling=True`` honored;
    # easy modes (``optimization_target="throughput"`` / ``"load"``)
    # silently force it back to False during config validation.
    return PlannerConfig(
        mode="agg",
        enable_load_scaling=True,
        enable_throughput_scaling=True,
        optimization_target="sla",
        served_model_name="test",
    )


def test_initial_tick_with_throughput_scaling_enabled_does_not_attribute_error():
    """``initial_tick`` used to read the non-existent
    ``throughput_adjustment_interval`` attribute (canonical name has a
    ``_seconds`` suffix; the short form is only a validation alias, not
    an attribute accessor in Pydantic v2). Pre-fix this branch raised
    ``AttributeError`` and crashed planner startup whenever
    ``enable_throughput_scaling`` was True.
    """
    config = _agg_config_throughput_on()
    # Sanity guard: if the validator ever changes and silently flips
    # this off, the test would pass for the wrong reason (the buggy
    # branch is short-circuited at line 340 ``if enable_throughput_scaling``).
    assert config.enable_throughput_scaling is True

    adapter = OrchestratorEngineAdapter(config, _caps())
    tick = adapter.initial_tick(start_s=0.0)
    assert isinstance(tick, ScheduledTick)
    # First tick is whichever cadence is shorter. We don't pin the exact
    # value here — defaults move between SLA presets — only that we
    # got past the buggy attribute read.
    assert tick.at_s > 0.0
    assert tick.run_load_scaling or tick.run_throughput_scaling


def test_merge_tolerance_matches_psm_500ms_window():
    """``_MERGE_TOLERANCE_S`` must be the PSM 500ms wiggle-room, not a
    float epsilon. Cadence advance anchors on ``tick_input.now_s``, so
    after a single tick the load and throughput schedules drift apart
    by however much wall-clock latency the tick took (typically a few
    ms). With ``1e-9`` tolerance such ticks fail to merge and the
    planner pays 2x scheduler overhead — PSM merges them into one.
    """
    adapter = OrchestratorEngineAdapter(_agg_config_throughput_on(), _caps())
    # Simulate cadences that are nearly coincident but offset by ~10ms
    # of wall-clock latency — well inside the 500ms PSM merge window.
    adapter._next_load_s = 180.010
    adapter._next_throughput_s = 180.0
    tick = adapter._compute_next_scheduled_tick()
    assert tick.run_load_scaling, "load cadence within 500ms must merge"
    assert tick.run_throughput_scaling, "throughput cadence within 500ms must merge"
    assert tick.at_s == pytest.approx(180.0, abs=1e-9)
