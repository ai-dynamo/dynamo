# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scorecard — turn one match's report into the Arena's ranking metrics.

The headline axis is **Efficiency = goodput / average allocated GPU**; we also compute
**SLO quality** (goodput rate) and **stability** (oscillation count). Goodput is
defined and computed EXACTLY as AIPerf defines it, so offline-sim numbers are
directly comparable to the online (AIPerf-driven) backend:

  goodput = (# requests meeting ALL SLO constraints jointly) / benchmark_duration_s

A request is "good" iff, for every non-null constraint in the profile, the
per-request value exists and is <= the threshold (inclusive). ITL is the
per-request inter-token latency AIPerf uses: ``(e2e - ttft) / (osl - 1)``,
undefined (→ not good) when ``osl < 2``. Comparing ms-vs-ms is identical to
AIPerf's ns-vs-ns. (Refs: aiperf good_request_count_metric.py / goodput_metric.py.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

SCORECARD_SCHEMA_VERSION = 2

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

    Falls back to ``(e2e - ttft) / (output_length - 1)`` only for legacy records
    lacking the field; that form is inexact under output clamping.
    """
    if "itl_ms" in rec:
        return rec["itl_ms"]
    ttft = rec.get("ttft_ms")
    e2e = rec.get("e2e_latency_ms")
    osl = rec.get("output_length") or 0
    if ttft is None or e2e is None or osl < 2:
        return None
    return (e2e - ttft) / (osl - 1)


def request_is_good(rec: dict[str, Any], profile: SLOProfile) -> bool:
    """True iff the request jointly meets every constraint in the profile."""
    if not profile.has_constraints:
        return False  # AIPerf: no SLOs → not counted as good
    terminal_status = rec.get("terminal_status")
    if terminal_status is None:
        terminal_status = rec.get("status")
    if terminal_status is not None and terminal_status != "completed":
        return False
    if profile.ttft_ms is not None:
        ttft = rec.get("ttft_ms")
        if ttft is None or ttft > profile.ttft_ms:
            return False
    if profile.e2e_ms is not None:
        e2e = rec.get("e2e_latency_ms")
        if e2e is None or e2e > profile.e2e_ms:
            return False
    if profile.itl_ms is not None:
        itl = request_itl_ms(rec)
        if itl is None or itl > profile.itl_ms:
            return False
    return True


def goodput_for_profile(
    per_request: list[dict[str, Any]],
    duration_s: float,
    attempted_requests: int,
    profile: SLOProfile,
) -> dict[str, float]:
    """Goodput stats for one SLO profile (AIPerf-exact).

    ``attempted_requests`` is the denominator for ``good_rate`` — AIPerf's
    good_request_fraction divides by attempted (request_count + errors), so
    dropped/rejected traffic correctly counts against the rate.
    """
    good = sum(1 for r in per_request if request_is_good(r, profile))
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
) -> dict[str, Any]:
    """Compute the full Arena scorecard for one match report.

    ``report`` exposes Dynamo's canonical replay summary plus per-request rows
    and planner scaling events. Goodput normally uses per-request capture
    (``capture_per_request=True``). When capture is disabled, pass
    ``sla_profile`` = the single profile whose SLA was supplied to replay;
    goodput for that profile is then read from the mocker's in-Rust
    ``goodput_*`` fields. Other profiles remain unavailable because the
    in-Rust goodput is single-SLA.
    """
    tr = report.trace_report
    duration_s = (tr.get("duration_ms") or 0.0) / 1000.0
    completed = int(tr.get("completed_requests") or 0)
    # AIPerf good_request_fraction denominator = attempted requests; num_requests
    # (total arrivals) is the closest analog (the mocker has no error count).
    attempted = int(tr.get("num_requests") or completed)
    trace_gpu_hours = tr.get("gpu_hours")
    gpu_hours = float(
        trace_gpu_hours
        if trace_gpu_hours is not None
        else (getattr(report, "gpu_hours", 0.0) or 0.0)
    )
    benchmark_hours = duration_s / 3600.0
    average_gpus = average_gpu_count(gpu_hours, duration_s)
    per_request = getattr(report, "per_request", None)

    osc = oscillation_count(report.scaling_events)

    out: dict[str, Any] = {
        "scorecard_schema_version": SCORECARD_SCHEMA_VERSION,
        "completed_requests": completed,
        "duration_s": duration_s,
        "benchmark_hours": benchmark_hours,
        "gpu_hours": gpu_hours,
        "average_gpus": average_gpus,
        "request_throughput_rps": tr.get("request_throughput_rps"),
        # stability axis
        "oscillation_count": osc["total"],
        "oscillation_by_component": {k: v for k, v in osc.items() if k != "total"},
        "scale_events": len(report.scaling_events),
        # latency pass-throughs (context columns)
        "mean_ttft_ms": tr.get("mean_ttft_ms"),
        "p95_ttft_ms": tr.get("p95_ttft_ms"),
        "p99_ttft_ms": tr.get("p99_ttft_ms"),
        "mean_itl_ms": tr.get("mean_itl_ms"),
        "p95_e2e_latency_ms": tr.get("p95_e2e_latency_ms"),
        "goodput_available": per_request is not None,
        "profiles": {},
    }

    # In-Rust single-SLA goodput fallback (no per-request capture needed).
    trace_goodput_rps = tr.get("goodput_request_throughput_rps")
    trace_good_count = tr.get("goodput_completed_requests")

    for profile in profiles:
        if per_request is None:
            if (
                sla_profile is not None
                and profile == sla_profile
                and trace_goodput_rps is not None
            ):
                goodput_per_gpu = (
                    trace_goodput_rps / average_gpus
                    if average_gpus is not None
                    else None
                )
                out["goodput_available"] = True
                out["profiles"][profile.name] = {
                    "good_count": trace_good_count,
                    "goodput_rps": trace_goodput_rps,
                    "good_rate": (trace_good_count / attempted) if attempted else None,
                    "goodput_per_gpu": goodput_per_gpu,
                    "goodput_source": "in_rust_sla",
                }
            else:
                out["profiles"][profile.name] = {
                    "goodput_rps": None,
                    "good_rate": None,
                    "goodput_per_gpu": None,
                }
            continue
        g = goodput_for_profile(per_request, duration_s, attempted, profile)
        goodput_per_gpu = (
            g["goodput_rps"] / average_gpus if average_gpus is not None else None
        )
        out["profiles"][profile.name] = {
            "good_count": g["good_count"],
            "goodput_rps": g["goodput_rps"],
            "good_rate": g["good_rate"],
            # Headline efficiency axis: good requests/sec per average GPU.
            "goodput_per_gpu": goodput_per_gpu,
        }
    return out


__all__ = [
    "SCORECARD_SCHEMA_VERSION",
    "SLOProfile",
    "DEFAULT_PROFILES",
    "request_itl_ms",
    "request_is_good",
    "goodput_for_profile",
    "average_gpu_count",
    "oscillation_count",
    "scorecard",
]
