# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scorecard — turn one match's report into the Arena's ranking metrics.

The headline axis is **Efficiency = goodput / average allocated GPU**; we also compute
**SLO quality** (goodput rate) and **stability** (oscillation count). Goodput is
AIPerf-compatible by default; an explicit short-output ITL policy can instead
match the native replay summary:

  goodput = (# requests meeting ALL SLO constraints jointly) / benchmark_duration_s

A request is "good" iff, for every non-null constraint in the profile, the
per-request value exists and is <= the threshold (inclusive). ITL is the
per-request inter-token latency AIPerf uses: ``(e2e - ttft) / (osl - 1)``,
undefined when ``osl < 2`` (the default policy treats that as not good). Comparing ms-vs-ms is identical to
AIPerf's ns-vs-ns. (Refs: aiperf good_request_count_metric.py / goodput_metric.py.)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, Optional

from autoscaling_arena.measurement import (
    AllocationConfig,
    MeasurementWindow,
    window_gpu_hours,
)

SCORECARD_SCHEMA_VERSION = 3
ShortOutputITLPolicy = Literal["fail", "skip"]

# ArenaReplayResult is a structural dependency; keep it typed loosely so this
# module remains importable for inspection without the optional Dynamo build.


@dataclass(frozen=True)
class SLOProfile:
    """An SLO target set; goodput is scored against one profile at a time.

    A request meets the profile iff it satisfies every non-``None`` constraint.
    All three are latency caps in milliseconds (smaller-is-better, inclusive).
    """

    name: str
    ttft_ms: Optional[float] = None
    itl_ms: Optional[float] = None
    e2e_ms: Optional[float] = None

    @property
    def has_constraints(self) -> bool:
        return any(c is not None for c in (self.ttft_ms, self.itl_ms, self.e2e_ms))


# Default profiles for interactive chat, agentic work, and relaxed comparison.
DEFAULT_PROFILES: tuple[SLOProfile, ...] = (
    SLOProfile(name="interactive", ttft_ms=300.0, itl_ms=50.0),
    SLOProfile(name="agentic", e2e_ms=3000.0, itl_ms=200.0),
    SLOProfile(name="relaxed", ttft_ms=2000.0, itl_ms=50.0),
)


def request_itl_ms(rec: dict[str, Any]) -> Optional[float]:
    """Per-request inter-token latency, matching AIPerf's ``inter_token_latency``.

    Prefers the mocker's own ``itl_ms`` field, which already divides by the
    *actually generated* token count (= AIPerf's OSL-based denominator) — correct
    even when output is clamped below the requested length. ``None`` there means
    the request emitted < 2 tokens (ITL undefined → request cannot satisfy an ITL
    constraint), mirroring AIPerf's NoMetricValue.

    Falls back to ``(e2e - ttft) / (output_length - 1)`` for legacy records
    lacking the field. Canonical replay records distinguish actual
    ``output_length`` from ``requested_output_length``; always use the former.
    """
    if "itl_ms" in rec:
        return rec["itl_ms"]
    ttft = rec.get("ttft_ms")
    e2e = rec.get("e2e_latency_ms")
    osl = rec.get("output_length") or 0
    if ttft is None or e2e is None or osl < 2:
        return None
    return (e2e - ttft) / (osl - 1)


def _validate_short_output_itl(policy: ShortOutputITLPolicy) -> None:
    if policy not in ("fail", "skip"):
        raise ValueError("short_output_itl must be 'fail' or 'skip'")


def _within_limit(value: Any, limit: float) -> bool:
    return value is not None and isfinite(value) and 0 <= value <= limit


def request_is_good(
    rec: dict[str, Any],
    profile: SLOProfile,
    *,
    short_output_itl: ShortOutputITLPolicy = "fail",
) -> bool:
    """True iff the request jointly meets every constraint in the profile.

    ``skip`` waives ITL for completed requests with exactly one emitted token,
    matching native replay. Zero emitted tokens or unknown token counts never
    waive a token-latency constraint. Other missing latencies always fail.
    """
    _validate_short_output_itl(short_output_itl)
    if not profile.has_constraints:
        return False  # AIPerf: no SLOs → not counted as good
    terminal_status = rec.get("terminal_status")
    if terminal_status is None:
        terminal_status = rec.get("status")
    if terminal_status is not None and terminal_status != "completed":
        return False
    if "first_admit_ms" in rec and rec["first_admit_ms"] is None:
        return False
    if profile.ttft_ms is not None:
        ttft = rec.get("ttft_ms")
        if not _within_limit(ttft, profile.ttft_ms):
            return False
    if profile.e2e_ms is not None:
        e2e = rec.get("e2e_latency_ms")
        # Native permits completed zero-output requests under an E2E-only SLA.
        # Their captured token-derived E2E field is absent; use terminal time.
        if e2e is None and rec.get("output_length") == 0:
            terminal_ms = rec.get("terminal_time_ms")
            arrival_ms = rec.get("arrival_time_ms")
            if terminal_ms is not None and arrival_ms is not None:
                e2e = terminal_ms - arrival_ms
        if not _within_limit(e2e, profile.e2e_ms):
            return False
    if profile.itl_ms is not None:
        if short_output_itl == "skip" and rec.get("output_length") == 1:
            return True
        if not _within_limit(request_itl_ms(rec), profile.itl_ms):
            return False
    return True


def goodput_for_profile(
    per_request: list[dict[str, Any]],
    duration_s: float,
    attempted_requests: int,
    profile: SLOProfile,
    *,
    short_output_itl: ShortOutputITLPolicy = "fail",
) -> dict[str, float]:
    """Goodput stats for one SLO profile and explicit short-output policy.

    ``attempted_requests`` is the denominator for ``good_rate`` — AIPerf's
    good_request_fraction divides by attempted (request_count + errors), so
    dropped/rejected traffic correctly counts against the rate.
    """
    _validate_short_output_itl(short_output_itl)
    good = sum(
        1
        for r in per_request
        if request_is_good(r, profile, short_output_itl=short_output_itl)
    )
    return {
        "good_count": float(good),
        # requests/sec meeting the SLO over the run window (AIPerf goodput).
        "goodput_rps": good / duration_s if duration_s > 0 else 0.0,
        # fraction of attempted requests that were good (AIPerf good_request_fraction).
        "good_rate": good / attempted_requests if attempted_requests > 0 else 0.0,
    }


def average_gpu_count(gpu_hours: float, duration_s: float) -> Optional[float]:
    """Return time-averaged allocated GPUs from cumulative GPU-hours.

    Dividing a rate such as goodput (requests/second) directly by GPU-hours
    makes the result depend on benchmark duration. The efficiency denominator
    is instead the average fleet size over that duration:

      average GPUs = reported GPU-hours / benchmark hours
    """
    benchmark_hours = duration_s / 3600.0
    if gpu_hours <= 0 or benchmark_hours <= 0:
        return None
    return gpu_hours / benchmark_hours


def oscillation_count(scaling_events: list[Any]) -> dict[str, int]:
    """Scale up↔down reversals, per component and total (the stability axis).

    A reversal is a scaling event whose direction flips relative to the
    previous event for the same component (up-after-down or down-after-up).
    """
    by_component: dict[str, list[str]] = {}
    for ev in scaling_events:
        by_component.setdefault(ev.component, []).append(ev.reason or "")
    reversals: dict[str, int] = {}
    for comp, reasons in by_component.items():
        count = 0
        last = None
        for r in reasons:
            if r in ("scale_up", "scale_down"):
                if last is not None and r != last:
                    count += 1
                last = r
        reversals[comp] = count
    reversals["total"] = sum(v for k, v in reversals.items() if k != "total")
    return reversals


def scorecard(
    report: Any,
    profiles: tuple[SLOProfile, ...] = DEFAULT_PROFILES,
    sla_profile: SLOProfile | None = None,
    *,
    short_output_itl: ShortOutputITLPolicy = "fail",
    measurement_window: MeasurementWindow | None = None,
    allocation: AllocationConfig | None = None,
    request_records: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute ranking metrics with explicit scoring and measurement semantics.

    Records may be supplied as a one-pass iterable (for example a JSONL reader)
    to avoid retaining request capture in memory. Otherwise ``report.per_request``
    is used. Without either, the native single-SLA summary is usable only when
    its profile and short-output policy match: native skips ITL for one-token
    outputs, whereas this function's default ``fail`` preserves AIPerf semantics.

    A measurement window selects arrivals in ``[start_s, end_s)`` and scores their
    eventual outcomes, including drain completions. Its rate denominator and
    allocated GPU-hours cover that same window. This requires complete request
    records and an allocation config. Context latency percentiles still describe
    the full replay and are explicitly labeled as such.
    """
    _validate_short_output_itl(short_output_itl)
    tr = report.trace_report
    replay_duration_s = (tr.get("duration_ms") or 0.0) / 1000.0
    completed = int(tr.get("completed_requests") or 0)
    attempted = int(tr.get("num_requests", completed))
    trace_gpu_hours = tr.get("gpu_hours")
    replay_gpu_hours = float(
        trace_gpu_hours
        if trace_gpu_hours is not None
        else (getattr(report, "gpu_hours", 0.0) or 0.0)
    )
    duration_s = replay_duration_s
    gpu_hours = replay_gpu_hours
    records = (
        request_records
        if request_records is not None
        else getattr(report, "per_request", None)
    )
    if measurement_window is not None:
        if records is None or allocation is None:
            raise ValueError(
                "measurement window requires request records and allocation"
            )
        planner = getattr(report, "planner", None)
        operations = getattr(planner, "lifecycle_operations", None)
        if operations is None and not allocation.allow_idle_tail:
            raise ValueError(
                "measurement window requires captured lifecycle operations"
            )
        gpu_hours = window_gpu_hours(
            operations or [],
            allocation,
            measurement_window,
            replay_duration_s=replay_duration_s,
        )
        duration_s = measurement_window.duration_s

    good_counts = [0] * len(profiles)
    if records is not None:
        selected_count = 0
        selected_completed = 0
        total_records = 0
        for rec in records:
            total_records += 1
            if measurement_window is not None:
                arrival_ms = rec.get("arrival_time_ms")
                if arrival_ms is None or not isfinite(arrival_ms):
                    raise ValueError(
                        "measurement window requires finite arrival_time_ms"
                    )
                if not (
                    measurement_window.start_s
                    <= arrival_ms / 1000.0
                    < measurement_window.end_s
                ):
                    continue
            selected_count += 1
            status = rec.get("terminal_status", rec.get("status"))
            if status is None or status == "completed":
                selected_completed += 1
            for index, profile in enumerate(profiles):
                good_counts[index] += request_is_good(
                    rec, profile, short_output_itl=short_output_itl
                )
        if total_records != attempted:
            raise ValueError(
                "scoring requires one record per attempted request "
                f"(received {total_records}, expected {attempted})"
            )
        if measurement_window is not None:
            attempted = selected_count
            completed = selected_completed

    benchmark_hours = duration_s / 3600.0
    average_gpus = average_gpu_count(gpu_hours, duration_s)
    osc = oscillation_count(report.scaling_events)
    out: dict[str, Any] = {
        "scorecard_schema_version": SCORECARD_SCHEMA_VERSION,
        "short_output_itl": short_output_itl,
        "measurement": {
            "basis": "arrival_window" if measurement_window else "full_replay",
            "start_s": measurement_window.start_s if measurement_window else 0.0,
            "end_s": measurement_window.end_s if measurement_window else duration_s,
            "completion_policy": "include_drain",
            "gpu_hours_basis": "lifecycle_window"
            if measurement_window
            else "native_replay",
            "allow_idle_tail": bool(measurement_window and allocation.allow_idle_tail),
        },
        "completed_requests": completed,
        "attempted_requests": attempted,
        "duration_s": duration_s,
        "replay_duration_s": replay_duration_s,
        "replay_gpu_hours": replay_gpu_hours,
        "benchmark_hours": benchmark_hours,
        "gpu_hours": gpu_hours,
        "average_gpus": average_gpus,
        "request_throughput_rps": (
            completed / duration_s
            if measurement_window
            else tr.get("request_throughput_rps")
        ),
        "oscillation_count": osc["total"],
        "oscillation_by_component": {k: v for k, v in osc.items() if k != "total"},
        "scale_events": len(report.scaling_events),
        "context_metrics_basis": "full_replay",
        "mean_ttft_ms": tr.get("mean_ttft_ms"),
        "p95_ttft_ms": tr.get("p95_ttft_ms"),
        "p99_ttft_ms": tr.get("p99_ttft_ms"),
        "mean_itl_ms": tr.get("mean_itl_ms"),
        "p95_e2e_latency_ms": tr.get("p95_e2e_latency_ms"),
        "goodput_available": records is not None,
        "profiles": {},
    }
    trace_good_count = tr.get("goodput_completed_requests")
    for index, profile in enumerate(profiles):
        if records is None:
            if not (
                sla_profile is not None
                and profile == sla_profile
                and profile.has_constraints
                and trace_good_count is not None
                and (profile.itl_ms is None or short_output_itl == "skip")
            ):
                out["profiles"][profile.name] = {
                    "goodput_rps": None,
                    "good_rate": None,
                    "goodput_per_gpu": None,
                    "unavailable_reason": (
                        "native_summary_skips_short_output_itl"
                        if profile.itl_ms is not None and short_output_itl == "fail"
                        else "matching_native_sla_summary_required"
                    ),
                }
                continue
            out["goodput_available"] = True
            good = trace_good_count
        else:
            good = good_counts[index]
        goodput_rps = good / duration_s if duration_s > 0 else 0.0
        out["profiles"][profile.name] = {
            "good_count": float(good),
            "goodput_rps": goodput_rps,
            "good_rate": good / attempted if attempted > 0 else 0.0,
            "goodput_per_gpu": (
                goodput_rps / average_gpus if average_gpus is not None else None
            ),
            "goodput_source": "request_records"
            if records is not None
            else "in_rust_sla",
        }
    return out


__all__ = [
    "SCORECARD_SCHEMA_VERSION",
    "ShortOutputITLPolicy",
    "SLOProfile",
    "DEFAULT_PROFILES",
    "request_itl_ms",
    "request_is_good",
    "goodput_for_profile",
    "average_gpu_count",
    "oscillation_count",
    "scorecard",
]
