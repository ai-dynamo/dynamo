# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A recording stand-in for the planner retune controller.

The engine-adapter and replay-host tests install ``StubController`` as
``dynamo.planner.retune.controller`` so they cover only what the hosts do
with the controller. This is a plain module, not a conftest, so the stub
reaches only the tests that import it.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

from dynamo.planner.config.planner_config import PlannerConfig, RetuneEngineSpec
from dynamo.planner.core.types import (
    EngineCapabilities,
    RetuneDecision,
    RetuneInputs,
    WorkerCapabilities,
)

CONTROLLER_MODULE = "dynamo.planner.retune.controller"
ROUTER_MAP = {
    "aggregated": {
        "decode_active_request_weight": 0.0,
        "disk_cache_hit_weight": 0.25,
        "host_cache_hit_weight": 0.75,
        "overlap_score_credit": 1.0,
        "overlap_score_credit_decay": 0.0,
        "prefill_load_scale": 1.0,
        "router_temperature": 0.0,
        "shared_cache_multiplier": 0.0,
    }
}


@dataclass(frozen=True)
class StubTickResult:
    decisions: list[RetuneDecision] = field(default_factory=list)
    router_parameters: dict[str, dict[str, float]] | None = None
    audit_events: list[str] = field(default_factory=list)


class StubController:
    """Records the host's calls; returns one queued tick result per tick."""

    instances: list[StubController] = []

    def __init__(self, config, inputs, capabilities, clock, scale_interval_s):
        self.args = (config, inputs, capabilities, clock, scale_interval_s)
        self.config = config
        self.started: list[float] = []
        self.ticks: list[tuple[float, Any]] = []
        # Searched planner values as the controller saw them on each tick.
        self.seen_planner_values: list[dict[str, Any]] = []
        self.results: list[StubTickResult] = []
        self.closed = 0
        self.close_error: Exception | None = None
        StubController.instances.append(self)

    def start(self, start_s: float) -> None:
        self.started.append(start_s)

    async def on_tick(self, now_s, worker_counts):
        self.ticks.append((now_s, worker_counts))
        self.seen_planner_values.append(
            {
                "load_scaling_down_sensitivity": self.config.load_scaling_down_sensitivity,
                "load_predictor": self.config.load_predictor,
            }
        )
        return self.results.pop(0) if self.results else StubTickResult()

    def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


def install_stub_controller(monkeypatch) -> type[StubController]:
    """Install ``StubController`` as ``dynamo.planner.retune.controller``."""

    module = types.ModuleType(CONTROLLER_MODULE)
    module.RetuneController = StubController  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, CONTROLLER_MODULE, module)
    StubController.instances = []
    return StubController


def caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1, max_num_batched_tokens=2048, max_kv_tokens=16384
        )
    )


def retune_only_config(**overrides) -> PlannerConfig:
    return PlannerConfig(
        mode="agg",
        enable_load_scaling=False,
        enable_throughput_scaling=False,
        retune={
            "interval_seconds": 60,
            "router": {
                "overlap_score_credit": 1.0,
                "prefill_load_scale": 1.0,
                "temperature": {"choices": [0.0, 0.5]},
            },
        },
        **overrides,
    )


def retune_inputs() -> RetuneInputs:
    return RetuneInputs(engine=RetuneEngineSpec(), trace_paths=("trace.jsonl",))
