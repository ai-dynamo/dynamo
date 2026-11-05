# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402
# Optional-dependency preflight must run before the simulation imports.

"""``planner.retune``, ``load_predictor_warmup_trace`` and ``planner.advisory``
in ``aisimulate predict``."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

from aisimulate.config_adapter import (
    PredictionAdapterContext,
    RecommendationAdapterContext,
)
from aisimulate.sweeper.provider import SweepContext

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.retune.window import read_trace_history
from dynamo.planner.simulation import create_provider
from dynamo.replay import api

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]

# AISimulate recommend syntax with public names; credit and scale are pinned.
_RETUNE: dict[str, Any] = {
    "interval_seconds": 120,
    "grace_seconds": 1,
    "router": {
        "overlap_score_credit": 1.0,
        "prefill_load_scale": 1.0,
        "temperature": {"choices": [0.0, 100.0]},
    },
}
_RETUNE_ONLY: dict[str, Any] = {
    "policy": "enabled",
    "enable_load_scaling": False,
    "enable_throughput_scaling": False,
}
_SLA = {"ttft_ms": 60.0, "itl_ms": 100.0}
_BASE_MS = 1_700_000_000_000


def _context(*, sla: dict | None = None) -> PredictionAdapterContext:
    return PredictionAdapterContext(
        engine={
            "mode": "aggregated",
            "workers": {"aggregated": {"parallelism": {"tensor": 1}}},
        },
        traffic={
            "source": {"type": "trace", "format": "dynamo", "paths": ["t.jsonl"]},
            "load": {"type": "trace_timestamps"},
        },
        evaluation={"sla": sla} if sla is not None else {},
    )


def _hook_config(spec) -> dict[str, Any]:
    (hook,) = spec.runtime_hooks
    return hook.config["planner_config"]


def _host_config(spec) -> PlannerConfig:
    # The replay host parses the hook payload from JSON.
    return PlannerConfig.from_config_arg(json.dumps(_hook_config(spec)))


def test_retune_reaches_the_replay_host_as_written() -> None:
    provider = create_provider()
    spec = provider.compile_prediction(
        {**_RETUNE_ONLY, "retune": _RETUNE}, _context(sla=_SLA)
    )

    payload = _hook_config(spec)
    assert payload["enable_load_scaling"] is False
    assert payload["enable_throughput_scaling"] is False
    # As written, not a defaults-filled dump; the scoring SLA defaults to
    # evaluation.sla.
    assert payload["retune"] == {**_RETUNE, "sla_ttft_ms": 60.0, "sla_itl_ms": 100.0}
    config = _host_config(spec)
    assert {"enable_load_scaling", "enable_throughput_scaling"} <= (
        config.model_fields_set
    )
    assert config.retune is not None and not config.scaling_enabled()
    assert config.retune.scoring_sla(1.0, 1.0) == (60.0, 100.0)
    # A retune SLA of its own is kept.
    explicit = {**_RETUNE, "sla_ttft_ms": 30.0, "sla_itl_ms": None}
    kept = provider.compile_prediction(
        {**_RETUNE_ONLY, "retune": explicit}, _context(sla=_SLA)
    )
    assert _hook_config(kept)["retune"] == explicit


def _write_trace(path: Path, *, count: int, start_ms: int, prefix: str) -> None:
    rows = [
        {
            "schema": "dynamo.request.trace.v1",
            "event_type": "request_end",
            "event_time_unix_ms": start_ms + index * 100 + 10,
            "request": {
                "request_id": f"{prefix}-{index}",
                "request_received_ms": start_ms + index * 100,
                "output_tokens": 2,
                "replay": {
                    "trace_block_size": 64,
                    "input_length": 64,
                    "input_sequence_hashes": [index],
                },
            },
        }
        for index in range(count)
    ]
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if path.name.endswith(".gz"):
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)


@pytest.mark.parametrize("warmup", ["history", "history/*.jsonl.gz"])
def test_warmup_directory_or_glob_is_the_retune_history(
    warmup, tmp_path, monkeypatch
) -> None:
    (tmp_path / "history").mkdir()
    for shard in (0, 1):
        _write_trace(
            tmp_path / f"history/{shard}.jsonl.gz",
            count=3,
            start_ms=_BASE_MS + shard * 1_000,
            prefix=f"history-{shard}",
        )
    trace = tmp_path / "trace.jsonl"
    _write_trace(trace, count=3, start_ms=_BASE_MS + 10_000, prefix="request")
    monkeypatch.chdir(tmp_path)
    planner = {
        "policy": "enabled",
        "target": "sla",
        "enable_throughput_scaling": True,
        "enable_load_scaling": False,
        "retune": {**_RETUNE, "planner": "default"},
    }

    # Absolute, against the working directory, with the glob kept.
    spec = create_provider().compile_prediction(
        {**planner, "load_predictor_warmup_trace": warmup}, _context(sla=_SLA)
    )
    config = _host_config(spec)
    assert config.load_predictor_warmup_trace == str(tmp_path / warmup)
    replay_kwargs: dict[str, Any] = {
        "trace_files": [trace],
        "trace_format": "dynamo",
        "replay_concurrency": None,
        "agentic_lanes": None,
        "agentic_snapshot": None,
        "agentic_warmup": False,
        "agentic_profile": None,
        "max_sim_time_ms": None,
        "extra_engine_args": None,
        "prefill_engine_args": None,
        "decode_engine_args": None,
        "router_mode": "kv_router",
        "router_config": None,
        "ais_perf_config": None,
        "performance_model_metadata": None,
        "model_name": None,
        "kv_event_lag_ms": None,
        "benchmark_granularity": 8,
        "arrival_speedup_ratio": 1.0,
    }
    inputs = api._retune_inputs(config, **replay_kwargs)
    assert inputs.history_paths == (str(tmp_path / warmup),)
    # Both shards are read as history before the first replayed request.
    assert (
        read_trace_history(
            inputs.history_paths, -math.inf, inputs.trace_origin_ms, math.inf
        )
        == 6
    )

    # A glob or a file that matches nothing is reported when the replay
    # reads it.
    for missing in ("missing/*.jsonl.gz", "missing.jsonl"):
        spec = create_provider().compile_prediction(
            {**planner, "load_predictor_warmup_trace": missing}, _context(sla=_SLA)
        )
        with pytest.raises(ValueError, match="No warmup trace files matched"):
            api._retune_inputs(_host_config(spec), **replay_kwargs)


def test_prediction_only_fields_need_an_enabled_policy_and_predict() -> None:
    provider = create_provider()
    for advisory in (True, False):
        spec = provider.compile_prediction(
            {"policy": "enabled", "advisory": advisory}, _context()
        )
        assert _hook_config(spec)["advisory"] is advisory
        assert _host_config(spec).advisory is advisory
    # A disabled planner makes no recommendations and runs no retune;
    # advisory: false is a no-op.
    for config, match in (
        ({"retune": _RETUNE}, "planner.retune requires planner.policy: enabled"),
        ({"advisory": True}, "planner.advisory requires planner.policy: enabled"),
    ):
        with pytest.raises(ValueError, match=match):
            provider.compile_prediction(config, _context())
    spec = provider.compile_prediction({"advisory": False}, _context())
    assert spec.config == {"policy": "disabled"}
    # Each recommendation candidate replays a static planner.
    context = RecommendationAdapterContext(
        engine={},
        traffic={},
        evaluation={},
        optimization={},
        sweep=SweepContext(
            core_search_space={"deployment_mode": ["agg"]},
            workload={"trace_path": None, "isl": 512, "osl": 128},
            goal={"target": "throughput"},
            show_progress=False,
        ),
    )
    for field in ({"retune": _RETUNE}, {"advisory": False}):
        with pytest.raises(ValueError, match="not supported by aisimulate recommend"):
            provider.compile_recommendation({"policy": "enabled", **field}, context)
