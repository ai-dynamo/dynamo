# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Scenario reports for fault tolerance testing.

Reports have:
- generate(ctx): Create the report after checks pass
- description: Human-readable description

Reports are run after all checks pass and can generate
additional artifacts (HTML reports, metrics summaries, etc.)
"""

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from tests.fault_tolerance.deploy.scenario import ScenarioContext


# =============================================================================
# Report Base Class
# =============================================================================


@dataclass
class Report(ABC):
    """Base class for report generation.

    Reports are run after checks pass and can generate
    additional artifacts (HTML reports, metrics summaries, etc.)
    """

    @abstractmethod
    def generate(self, ctx: "ScenarioContext") -> None:
        """Generate the report."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of the report."""
        pass


# =============================================================================
# Fault Tolerance Report
# =============================================================================


@dataclass
class FaultToleranceReport(Report):
    """Render a markdown table of pre/post-fault metrics, one row per fault.

    Reads the per-request aiperf JSONL produced by ``StartLoad`` and slices
    it on each fault-injection event's ``started_at`` timestamp (recorded
    by the scenario runner). Mirrors the legacy fault-tolerance report:

        | Failure | Success Before | Failed Before | Success After | Failed After | Latency Before | Latency After |

    A baseline ``none`` row is always emitted; one additional row per
    ``DeletePod`` / ``TerminateProcess`` / ``RollingUpgrade`` event.
    """

    save_to_file: bool = True

    def generate(self, ctx: "ScenarioContext") -> None:
        # Local import avoids a circular dependency at module load.
        from tests.fault_tolerance.deploy.events import (
            DeletePod,
            RollingUpgrade,
            StartLoad,
            TerminateProcess,
        )

        FAULT_TYPES = (DeletePod, TerminateProcess, RollingUpgrade)

        load_events = [e for e in ctx.events if isinstance(e, StartLoad)]
        if not load_events:
            ctx.logger.info("FaultToleranceReport: no StartLoad in scenario; skipping")
            return

        # Most fault scenarios run a single load; if there are several, use
        # the first as the per-request source. (Per-load reports could be a
        # future extension.) Fall back to the deterministic
        # ``<log_dir>/load/`` path because StopLoad clears the
        # ``_managed_load`` reference before reports run.
        load = load_events[0]
        ml = getattr(load, "_managed_load", None)
        local_dir = (
            getattr(ml, "local_output_dir", None) if ml else None
        ) or os.path.join(ctx.log_dir, "load")
        jsonl_path = os.path.join(local_dir, "profile_export.jsonl")
        if not os.path.exists(jsonl_path):
            ctx.logger.warning(
                f"FaultToleranceReport: per-request log {jsonl_path} not found"
            )
            return

        records: list[dict[str, Any]] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        # Startup time is a property of the deployment, not of any
        # particular fault — it's the same value on every row. The
        # deployment reference has been cleared by the time reports run,
        # so we read the value from ctx (cached pre-cleanup).
        startup_s = getattr(ctx, "startup_seconds", None)

        rows = [self._row("none", records, fault_ns=None, startup_s=startup_s)]
        for fe in ctx.events:
            if not isinstance(fe, FAULT_TYPES):
                continue
            started = getattr(fe, "started_at", None)
            if started is None:
                continue
            fault_ns = int(started.timestamp() * 1_000_000_000)
            label = type(fe).__name__
            services = getattr(fe, "services", None)
            if services:
                label = f"{label}({','.join(services)})"
            rows.append(
                self._row(label, records, fault_ns=fault_ns, startup_s=startup_s)
            )

        table = self._render(rows)
        ctx.logger.info("\n" + table)

        if self.save_to_file:
            out = os.path.join(ctx.log_dir, "fault_tolerance_report.md")
            try:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w") as f:
                    f.write(table + "\n")
                ctx.logger.info(f"FaultToleranceReport written to {out}")
            except OSError as e:
                ctx.logger.warning(f"FaultToleranceReport save failed: {e}")

    @property
    def description(self) -> str:
        return "Fault-tolerance summary table (pre/post per fault event)"

    @staticmethod
    def _row(
        label: str,
        records: list[dict],
        fault_ns: Optional[int],
        startup_s: Optional[float] = None,
    ) -> dict[str, Any]:
        """Compute a single table row.

        ``fault_ns=None`` means "no fault" — counts and latency span the
        whole load window and post-fault columns are reported as ``N/A``.
        """
        succ_b = fail_b = succ_a = fail_a = 0
        lat_b: list[float] = []
        lat_a: list[float] = []
        for r in records:
            md = r.get("metadata", {})
            mt = r.get("metrics", {})
            end_ns = md.get("request_end_ns", 0)
            # A request is a failure if aiperf cancelled it OR if it
            # finished without producing a latency metric (which is what
            # TimeoutError / connection-reset records look like in the
            # JSONL — `was_cancelled` stays False but `metrics` is empty).
            lat_record = mt.get("request_latency")
            failed = bool(md.get("was_cancelled", False)) or not lat_record
            lat = (lat_record or {}).get("value", 0.0)
            before = fault_ns is None or end_ns < fault_ns
            if before:
                if failed:
                    fail_b += 1
                else:
                    succ_b += 1
                    lat_b.append(lat)
            else:
                if failed:
                    fail_a += 1
                else:
                    succ_a += 1
                    lat_a.append(lat)

        def _avg(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        return {
            "failure": label,
            "startup_s": "N/A" if startup_s is None else startup_s,
            "success_before": succ_b,
            "failed_before": fail_b,
            "success_after": "N/A" if fault_ns is None else succ_a,
            "failed_after": "N/A" if fault_ns is None else fail_a,
            "latency_before_ms": _avg(lat_b),
            "latency_after_ms": ("N/A" if fault_ns is None else _avg(lat_a)),
        }

    @staticmethod
    def _render(rows: list[dict[str, Any]]) -> str:
        header = (
            "| Failure | Startup (s) | Success Before | Failed Before "
            "| Success After | Failed After | Latency Before (ms) | Latency After (ms) |"
        )
        sep = "|---|---|---|---|---|---|---|---|"
        out = [header, sep]
        for r in rows:
            lb = f"{r['latency_before_ms']:.2f}"
            la = (
                r["latency_after_ms"]
                if isinstance(r["latency_after_ms"], str)
                else f"{r['latency_after_ms']:.2f}"
            )
            startup = (
                r["startup_s"]
                if isinstance(r["startup_s"], str)
                else f"{r['startup_s']:.2f}"
            )
            out.append(
                f"| {r['failure']} | {startup} | {r['success_before']} | {r['failed_before']} "
                f"| {r['success_after']} | {r['failed_after']} | {lb} | {la} |"
            )
        return "\n".join(out)
