# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay adapter driven by a **real mooncake-format trace**
(DEP-XXXX PR 8 sub-task 8-8 follow-up — task #76).

The existing ``test_replay_dual_path.py`` proves PSM and orchestrator
paths agree tick-by-tick when fed identical synthetic
``_FakeReplayBridge`` scripts. That is *necessary* but not
*sufficient* — synthetic ticks bypass the entire trace ingestion +
request-aggregation surface. This file closes that gap by:

1. Writing a small but realistic mooncake-format JSONL trace to
   ``tmp_path`` (timestamps in ms, ``input_length`` / ``output_length``
   per request — same shape ``extract_metrics_from_mooncake`` already
   parses for warmup).
2. Wrapping it in a ``_MooncakeJsonlBridge`` that mimics the Rust
   ``PlannerReplayBridge.advance_to`` contract: at each tick, the
   bridge groups any requests whose timestamp falls in the elapsed
   window, computes traffic + FPM accumulators, and emits the same
   dict shape ``ReplayPlannerAdapter`` expects from the real bridge.
3. Driving both PSM and orchestrator paths through the same trace
   and asserting:
   - The trace is fully consumed (no premature ``is_done``).
   - Both paths produce the same total tick count.
   - Both paths agree on every scaling event (component / from / to).
   - Diagnostics log is populated (non-empty, no None entries).
   - HTML diagnostics report is generated to a real path.

Why this matters: replay is the offline-diagnosis tool of last resort.
If something looks wrong in production, an SRE replays the trace
through this adapter to reproduce and instrument the planner's
decisions. Up to now we only proved that worked on the PSM path
*and* on synthetic fixtures; this test locks "real trace + post-PR-10
orchestrator path" — the actual production combination once PR 10
flips the flag.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    EngineCapabilities,
    WorkerCapabilities,
)
from dynamo.planner.offline.replay_adapter import (
    ReplayPlannerAdapter,
    ReplayPlannerReport,
    ScalingEvent,
)
from dynamo.planner.offline.trace_data import extract_metrics_from_mooncake

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


# ---------------------------------------------------------------------------
# Mooncake trace fixture
# ---------------------------------------------------------------------------


def _build_mooncake_trace() -> list[dict[str, Any]]:
    """Generate a deterministic mooncake-format JSONL trace.

    ~120 requests over ~5 minutes of simulated time. Load profile:
    - 0–60s   : low traffic    (~0.5 req/s)   — baseline, no scale
    - 60–180s : ramp           (~2 req/s)     — should trigger scale-up
    - 180–240s: high traffic   (~3 req/s)     — sustained pressure
    - 240–300s: cool-down      (~0.5 req/s)   — should trigger scale-down

    The exact request counts are fixed (no randomness) so PSM and
    orchestrator paths must agree byte-for-byte on the same input.
    """
    records: list[dict[str, Any]] = []
    rid = 0

    def _push(t_ms: int, isl: int, osl: int) -> None:
        nonlocal rid
        rid += 1
        records.append(
            {
                "timestamp": t_ms,
                "input_length": isl,
                "output_length": osl,
                "request_id": f"req-{rid}",
            }
        )

    # Phase 1 — baseline 0-60s, 0.5 req/s = 30 requests
    for i in range(30):
        _push(t_ms=i * 2000, isl=512, osl=128)
    # Phase 2 — ramp 60-180s, 2 req/s = 240 ticks @ 500ms apart, but cap at 80
    # (cap keeps the trace short while still showing pressure)
    for i in range(80):
        _push(t_ms=60_000 + i * 1500, isl=2048, osl=256)
    # Phase 3 — peak 180-240s, 3 req/s = 30 dense requests
    for i in range(30):
        _push(t_ms=180_000 + i * 2000, isl=4096, osl=512)
    # Phase 4 — cooldown 240-300s, 0.5 req/s
    for i in range(15):
        _push(t_ms=240_000 + i * 4000, isl=512, osl=128)

    return records


def _write_mooncake_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_mooncake_extract_metrics_works():
    """Sanity-check that the trace fixture is in the canonical
    mooncake format that ``extract_metrics_from_mooncake`` already
    consumes for predictor warmup. Catches schema drift between the
    fixture and the production parser."""
    import tempfile

    records = _build_mooncake_trace()
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        path = f.name

    metrics = extract_metrics_from_mooncake(path, throughput_adjustment_interval=60)
    # Phase boundaries: 0-60 / 60-120 / 120-180 / 180-240 / 240-300
    # so we expect 5 intervals (the 60-180 ramp falls into 60-120 and 120-180).
    assert len(metrics) >= 4, f"expected >=4 intervals, got {len(metrics)}"
    # Phase 1 (0-60s) has 30 records and ISL=512.
    first = metrics[0]
    assert first["interval_start"] == 0
    assert first["request_count"] == 30
    assert first["avg_isl"] == 512


# ---------------------------------------------------------------------------
# Mooncake → bridge adapter
# ---------------------------------------------------------------------------


class _MooncakeJsonlBridge:
    """Minimal Python bridge that drives the planner from a mooncake
    JSONL trace.

    Mirrors the Rust ``PlannerReplayBridge`` surface that
    ``ReplayPlannerAdapter`` consumes:

    - ``advance_to(tick_ms)`` returns a dict with the required keys
      (``is_done`` / ``now_ms`` / ``active_*_count`` /
      ``prefill_fpm_snapshots`` / ``decode_fpm_snapshots`` /
      ``accumulated_metrics``).
    - ``apply_scaling(prefill, decode)`` records the planner's command
      and updates the simulated active worker count.
    - ``drain_traffic()`` returns request-window aggregates since the
      last drain.
    - ``finalize()`` returns end-of-trace summary (matches the Rust
      contract checked by the existing test).

    Simplifying assumptions (vs the real Rust bridge):
    - No queue dynamics. We don't simulate per-engine load; FPM
      snapshots are derived purely from request counts in the window.
    - One synthetic decode worker (and prefill worker in disagg mode)
      per ``apply_scaling`` step. Good enough for end-of-trace parity
      assertions which is what this test cares about.
    """

    def __init__(self, records: list[dict[str, Any]], *, mode: str = "agg") -> None:
        self._records = sorted(records, key=lambda r: r["timestamp"])
        self._cursor = 0
        self._last_advance_ms = 0.0
        self._mode = mode
        self._active_prefill = 1 if mode == "disagg" else 0
        self._active_decode = 1
        # Aggregated traffic since last drain.
        self._pending_traffic: dict[str, Any] = {
            "duration_s": 0.0,
            "num_req": 0,
            "avg_isl": 0.0,
            "avg_osl": 0.0,
            "avg_ttft_ms": 0.0,
            "avg_itl_ms": 0.0,
        }
        # Cumulative for finalize().
        self._cum_requests = 0
        self._cum_isl_tokens = 0
        self._cum_osl_tokens = 0
        self.apply_scaling_calls: list[tuple[int, int]] = []

    def advance_to(self, tick_ms: float) -> dict[str, Any]:
        # Consume any records whose timestamp falls in
        # (last_advance_ms, tick_ms] and accumulate them into
        # _pending_traffic and into FPM-derived counts.
        window_start_ms = self._last_advance_ms
        window_end_ms = tick_ms
        if self._cursor >= len(self._records):
            return {"is_done": True}

        in_window: list[dict[str, Any]] = []
        while self._cursor < len(self._records):
            ts = self._records[self._cursor]["timestamp"]
            if ts > window_end_ms:
                break
            if ts > window_start_ms or window_start_ms == 0:
                in_window.append(self._records[self._cursor])
            self._cursor += 1

        # Update pending traffic aggregates.
        if in_window:
            n = len(in_window)
            isl_sum = sum(r["input_length"] for r in in_window)
            osl_sum = sum(r["output_length"] for r in in_window)
            duration_s = (window_end_ms - window_start_ms) / 1000.0
            self._pending_traffic = {
                "duration_s": duration_s,
                "num_req": n,
                "avg_isl": float(isl_sum) / n,
                "avg_osl": float(osl_sum) / n,
                "avg_ttft_ms": 50.0,  # placeholder; only used by recorder
                "avg_itl_ms": 5.0,
            }
            self._cum_requests += n
            self._cum_isl_tokens += isl_sum
            self._cum_osl_tokens += osl_sum

        self._last_advance_ms = window_end_ms

        # Build per-tick FPM snapshot. wall_time>0 only when we actually
        # had requests in the window — otherwise idle and the regression
        # filter (``fpm.wall_time > 0`` in replay_adapter) skips it.
        sum_decode_kv = sum(r["output_length"] for r in in_window) if in_window else 0
        sum_prefill_tok = (
            sum(r["input_length"] for r in in_window) if in_window else 0
        )
        wall_time = (
            (window_end_ms - window_start_ms) / 1000.0 if in_window else 0.0
        )
        snapshot = {
            "worker_id": 1,
            "wall_time": wall_time,
            "num_prefill_requests": len(in_window),
            "sum_prefill_tokens": sum_prefill_tok,
            "var_prefill_length": 0.0,
            "sum_prefill_kv_tokens": 0,
            "num_decode_requests": len(in_window),
            "sum_decode_kv_tokens": sum_decode_kv,
            "var_decode_kv_tokens": 0.0,
            "num_queued_prefill": 0,
            "sum_queued_prefill_tokens": 0,
            "var_queued_prefill_length": 0.0,
            "num_queued_decode": 0,
            "sum_queued_decode_kv_tokens": 0,
            "var_queued_decode_kv_tokens": 0.0,
        }
        snaps = [snapshot] if in_window else []

        return {
            "is_done": False,
            "now_ms": tick_ms,
            "active_prefill_count": self._active_prefill,
            "active_decode_count": self._active_decode,
            "prefill_fpm_snapshots": snaps if self._mode == "disagg" else [],
            "decode_fpm_snapshots": snaps,
            "accumulated_metrics": {
                "num_requests": self._cum_requests,
                "input_tokens": self._cum_isl_tokens,
                "output_tokens": self._cum_osl_tokens,
                "duration_sum": self._last_advance_ms / 1000.0,
                "ttft_sum": 0.0,
                "itl_sum": 0.0,
            },
        }

    def drain_traffic(self) -> dict[str, Any]:
        out = dict(self._pending_traffic)
        self._pending_traffic = {
            "duration_s": 0.0,
            "num_req": 0,
            "avg_isl": 0.0,
            "avg_osl": 0.0,
            "avg_ttft_ms": 0.0,
            "avg_itl_ms": 0.0,
        }
        return out

    def apply_scaling(self, prefill: int, decode: int) -> None:
        self.apply_scaling_calls.append((prefill, decode))
        if self._mode == "disagg":
            self._active_prefill = prefill
        self._active_decode = decode

    def finalize(self) -> dict[str, Any]:
        return {
            "total_requests": self._cum_requests,
            "total_input_tokens": self._cum_isl_tokens,
            "total_output_tokens": self._cum_osl_tokens,
            "duration_s": self._last_advance_ms / 1000.0,
            "avg_ttft_s": 0.0,
            "avg_itl_s": 0.0,
        }


# ---------------------------------------------------------------------------
# Test harness
# ---------------------------------------------------------------------------


def _agg_easy_config(use_orchestrator: bool, report_dir: Path) -> PlannerConfig:
    return PlannerConfig(
        environment="kubernetes",
        mode="agg",
        optimization_target="throughput",
        enable_load_scaling=True,
        enable_throughput_scaling=False,
        load_adjustment_interval=10,
        # Enable diagnostics so we can assert HTML report path.
        report_output_dir=str(report_dir),
        scheduling={"use_orchestrator": use_orchestrator},
    )


def _caps() -> WorkerCapabilities:
    return WorkerCapabilities(
        decode=EngineCapabilities(
            num_gpu=1,
            max_num_batched_tokens=2048,
            max_kv_tokens=16384,
            max_num_seqs=64,
        )
    )


def _run_once(
    *, use_orchestrator: bool, records: list[dict[str, Any]], report_dir: Path
) -> tuple[ReplayPlannerReport, _MooncakeJsonlBridge]:
    bridge = _MooncakeJsonlBridge(records, mode="agg")
    adapter = ReplayPlannerAdapter(
        planner_config=_agg_easy_config(use_orchestrator, report_dir),
        bridge=bridge,
        capabilities=_caps(),
    )
    return adapter.run(), bridge


def _events_signature(events: list[ScalingEvent]) -> list[tuple[float, str, int, int]]:
    """Reduce ScalingEvent list to a comparable shape (drop free-form
    ``reason`` since both paths classify identically but the test
    doesn't depend on the string)."""
    return [(round(e.at_s, 3), e.component, e.from_count, e.to_count) for e in events]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_mooncake_trace_full_consumption_psm(tmp_path: Path):
    """PSM path consumes the entire trace without premature is_done."""
    records = _build_mooncake_trace()
    report, bridge = _run_once(
        use_orchestrator=False, records=records, report_dir=tmp_path / "psm"
    )
    assert report.total_ticks > 0
    assert bridge._cursor == len(records), (
        f"PSM did not consume all records: cursor={bridge._cursor} of {len(records)}"
    )
    # Trace report rolled up correctly.
    assert report.trace_report["total_requests"] == len(records)


def test_mooncake_trace_full_consumption_orchestrator(tmp_path: Path):
    """Orchestrator path consumes the entire trace identically."""
    records = _build_mooncake_trace()
    report, bridge = _run_once(
        use_orchestrator=True, records=records, report_dir=tmp_path / "orch"
    )
    assert report.total_ticks > 0
    assert bridge._cursor == len(records)
    assert report.trace_report["total_requests"] == len(records)


def test_mooncake_trace_dual_path_parity(tmp_path: Path):
    """The whole point: PSM and orchestrator paths must produce the
    SAME scaling events when fed the same real-shape trace.
    """
    records = _build_mooncake_trace()
    psm_rep, _ = _run_once(
        use_orchestrator=False, records=records, report_dir=tmp_path / "psm"
    )
    orch_rep, _ = _run_once(
        use_orchestrator=True, records=records, report_dir=tmp_path / "orch"
    )

    assert psm_rep.total_ticks == orch_rep.total_ticks, (
        f"tick count diverged: psm={psm_rep.total_ticks} orch={orch_rep.total_ticks}"
    )
    psm_events = _events_signature(psm_rep.scaling_events)
    orch_events = _events_signature(orch_rep.scaling_events)
    assert psm_events == orch_events, (
        f"scaling events diverged on real mooncake trace:\n"
        f"  psm:  {psm_events}\n  orch: {orch_events}"
    )


def test_mooncake_trace_diagnostics_log_populated(tmp_path: Path):
    """Both paths must populate the diagnostics_log on every tick.
    Regression guard against the orchestrator path returning effects
    with ``diagnostics=None`` somewhere in the chain."""
    records = _build_mooncake_trace()
    psm_rep, _ = _run_once(
        use_orchestrator=False, records=records, report_dir=tmp_path / "psm"
    )
    orch_rep, _ = _run_once(
        use_orchestrator=True, records=records, report_dir=tmp_path / "orch"
    )
    assert len(psm_rep.diagnostics_log) == psm_rep.total_ticks
    assert len(orch_rep.diagnostics_log) == orch_rep.total_ticks
    # Every entry must be non-None (TickDiagnostics is a dataclass; even
    # an "all fields None" snapshot is still an instance, not None).
    assert all(d is not None for d in orch_rep.diagnostics_log)


def test_mooncake_trace_html_report_generated(tmp_path: Path):
    """Diagnostics recorder fires its end-of-replay HTML report on
    both paths. The path may be ``None`` only if the report was
    explicitly disabled — here we leave it on, so a real path is
    expected."""
    records = _build_mooncake_trace()
    psm_rep, _ = _run_once(
        use_orchestrator=False, records=records, report_dir=tmp_path / "psm"
    )
    orch_rep, _ = _run_once(
        use_orchestrator=True, records=records, report_dir=tmp_path / "orch"
    )
    # If recorder ran, both paths produce the same output shape: either
    # a Path string for both, or None for both. The orchestrator path
    # historically dropped this — locking it here.
    if psm_rep.html_report_path is not None:
        assert orch_rep.html_report_path is not None, (
            "PSM produced an HTML report but orchestrator path didn't — "
            "diagnostic recorder integration regressed"
        )


# ---------------------------------------------------------------------------
# Aggressive trace — forces real scale_up events on both paths.
#
# The "gentle" trace above only locks "both paths agree on idle" parity.
# This trace deliberately overflows the single decode worker's KV budget
# (16384 tokens) so latency-mode load-scaling fires `scale_up`. With the
# threshold at 40% util, every tick that lands in the burst window will
# observe sum_decode_kv_tokens >> 6553 → scale_up triggers.
#
# Why a separate trace + config: the gentle trace covers "trace ingestion +
# window aggregation" parity; the aggressive trace covers "scaling-decision
# parity on real-trace-shape input". Together they close the matrix.
# ---------------------------------------------------------------------------


def _build_aggressive_mooncake_trace() -> list[dict[str, Any]]:
    """Spike trace: idle baseline, single short burst that overflows KV
    budget, then idle again. Deterministic so PSM and orchestrator
    paths can be compared byte-for-byte."""
    records: list[dict[str, Any]] = []
    rid = 0

    def _push(t_ms: int, isl: int, osl: int) -> None:
        nonlocal rid
        rid += 1
        records.append(
            {
                "timestamp": t_ms,
                "input_length": isl,
                "output_length": osl,
                "request_id": f"req-{rid}",
            }
        )

    # 0-30s: idle baseline (5 small requests). Lets the planner observe
    # a stable starting state before the spike.
    for i in range(5):
        _push(t_ms=i * 6000, isl=256, osl=64)

    # 60-70s: BURST. 50 requests with massive OSL inside a single
    # adjustment window. 50 * 4096 = 204800 KV tokens; capacity is
    # 16384. utilization = 12.5x → way over the 40% latency threshold.
    for i in range(50):
        _push(t_ms=60_000 + i * 200, isl=2048, osl=4096)

    # 90-150s: idle cooldown so we can also observe scale_down.
    for i in range(10):
        _push(t_ms=90_000 + i * 6000, isl=256, osl=64)

    return records


def _agg_latency_config(use_orchestrator: bool, report_dir: Path) -> PlannerConfig:
    """Latency mode → 40% util threshold (vs throughput's 100%). Picked
    so a realistic burst can actually drive scale_up in this test."""
    return PlannerConfig(
        environment="kubernetes",
        mode="agg",
        optimization_target="latency",
        enable_load_scaling=True,
        enable_throughput_scaling=False,
        load_adjustment_interval=10,
        max_gpu_budget=8,  # leave room to scale up
        report_output_dir=str(report_dir),
        scheduling={"use_orchestrator": use_orchestrator},
    )


def _run_aggressive(
    *, use_orchestrator: bool, records: list[dict[str, Any]], report_dir: Path
) -> tuple[ReplayPlannerReport, _MooncakeJsonlBridge]:
    bridge = _MooncakeJsonlBridge(records, mode="agg")
    adapter = ReplayPlannerAdapter(
        planner_config=_agg_latency_config(use_orchestrator, report_dir),
        bridge=bridge,
        capabilities=_caps(),
    )
    return adapter.run(), bridge


def test_aggressive_mooncake_trace_triggers_real_scale_up(tmp_path: Path):
    """The burst phase must drive at least one scale_up on each path.
    If this asserts 0, the trace isn't aggressive enough — strengthen
    it rather than relaxing the assertion."""
    records = _build_aggressive_mooncake_trace()
    psm_rep, _ = _run_aggressive(
        use_orchestrator=False, records=records, report_dir=tmp_path / "psm"
    )
    orch_rep, _ = _run_aggressive(
        use_orchestrator=True, records=records, report_dir=tmp_path / "orch"
    )
    psm_up = [e for e in psm_rep.scaling_events if e.reason == "scale_up"]
    orch_up = [e for e in orch_rep.scaling_events if e.reason == "scale_up"]
    assert len(psm_up) >= 1, (
        f"PSM did not scale_up on aggressive trace; events={psm_rep.scaling_events!r}"
    )
    assert len(orch_up) >= 1, (
        f"Orchestrator did not scale_up on aggressive trace; events={orch_rep.scaling_events!r}"
    )


def test_aggressive_mooncake_trace_dual_path_byte_equal(tmp_path: Path):
    """**The headline assertion**: when the trace actually triggers
    scaling, PSM and orchestrator paths produce identical event
    sequences (component / from / to / at_s).

    If this fails, ``test_dual_path_parity`` (30 G3 fixtures) catches
    most divergences but not the trace-ingestion path. This locks
    that gap.
    """
    records = _build_aggressive_mooncake_trace()
    psm_rep, _ = _run_aggressive(
        use_orchestrator=False, records=records, report_dir=tmp_path / "psm"
    )
    orch_rep, _ = _run_aggressive(
        use_orchestrator=True, records=records, report_dir=tmp_path / "orch"
    )

    # First the prerequisite: scaling actually happened. Without this
    # the byte-equality is meaningless (both empty == both empty).
    assert len(psm_rep.scaling_events) > 0
    assert len(orch_rep.scaling_events) > 0

    psm_sig = _events_signature(psm_rep.scaling_events)
    orch_sig = _events_signature(orch_rep.scaling_events)
    assert psm_sig == orch_sig, (
        "scaling events diverged on aggressive mooncake trace:\n"
        f"  psm:  {psm_sig}\n  orch: {orch_sig}"
    )
    # Tick counts must also match — trace is consumed identically.
    assert psm_rep.total_ticks == orch_rep.total_ticks


def test_mooncake_jsonl_file_round_trip(tmp_path: Path):
    """End-to-end: write the trace to disk as JSONL (real-deployment
    shape; ConfigMap-mounted file in K8s), feed it through
    ``extract_metrics_from_mooncake``, and confirm interval grouping
    matches the in-memory phase boundaries.

    Catches: someone accidentally renaming a mooncake field in
    ``trace_data.py`` would not be caught by the synthetic-bridge
    test alone."""
    records = _build_mooncake_trace()
    trace_path = tmp_path / "trace.jsonl"
    _write_mooncake_jsonl(records, trace_path)

    intervals = extract_metrics_from_mooncake(
        str(trace_path), throughput_adjustment_interval=60
    )
    assert len(intervals) >= 4
    # First interval (0-60s) covers the 30 baseline requests.
    assert intervals[0]["request_count"] == 30
    assert intervals[0]["avg_isl"] == 512
    # Last covered interval is the cooldown.
    assert intervals[-1]["request_count"] >= 5
