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
            NetworkPartition,
            RollingUpgrade,
            StartLoad,
            TerminateProcess,
            WaitForRecovery,
        )

        FAULT_TYPES = (
            DeletePod,
            TerminateProcess,
            RollingUpgrade,
            NetworkPartition,
        )

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

        # Recovery time per fault: derived from the WaitForRecovery
        # event that immediately follows each fault in the events list.
        # WaitForRecovery's timed_execute brackets give wall-clock for
        # how long the deployment took to flip back to Ready after the
        # fault. NetworkPartition's transient (duration=N) is its own
        # recovery — partition starts and heals inside one execute call.
        recovery_by_event_id: dict[int, float] = {}
        for i, fe in enumerate(ctx.events):
            if not isinstance(fe, FAULT_TYPES):
                continue
            # Transient partition: recovery is the partition's own
            # in-execute window (the deployment never went non-Ready
            # since worker pods stayed running).
            if isinstance(fe, NetworkPartition) and fe.duration is not None:
                recovery_by_event_id[id(fe)] = float(fe.duration)
                continue
            for follower in ctx.events[i + 1 :]:
                if isinstance(follower, WaitForRecovery):
                    s = getattr(follower, "started_at", None)
                    e = getattr(follower, "ended_at", None)
                    if s and e:
                        recovery_by_event_id[id(fe)] = (e - s).total_seconds()
                    break
                if isinstance(follower, FAULT_TYPES):
                    # Another fault closes this fault's recovery window.
                    break

        rows = [
            self._row(
                "none",
                records,
                fault_ns=None,
                startup_s=startup_s,
                recovery_s=None,
            )
        ]
        for fe in ctx.events:
            if not isinstance(fe, FAULT_TYPES):
                continue
            started = getattr(fe, "started_at", None)
            if started is None:
                continue
            fault_ns = int(started.timestamp() * 1_000_000_000)
            label = type(fe).__name__
            services = getattr(fe, "services", None)
            source = getattr(fe, "source", None)
            target = getattr(fe, "target", None)
            if services:
                label = f"{label}({','.join(services)})"
            elif source and target:
                label = f"{label}({source}->{target})"
            rows.append(
                self._row(
                    label,
                    records,
                    fault_ns=fault_ns,
                    startup_s=startup_s,
                    recovery_s=recovery_by_event_id.get(id(fe)),
                )
            )

        # Render the report as three narrow sections — easier to read in
        # a terminal or slide deck than one wide eight-column table.
        ctx.logger.info("\n" + self._render_sections(rows, "fancy_grid"))

        if self.save_to_file:
            md_out = os.path.join(ctx.log_dir, "fault_tolerance_report.md")
            json_out = os.path.join(ctx.log_dir, "fault_tolerance_report.json")
            try:
                os.makedirs(os.path.dirname(md_out), exist_ok=True)
                with open(md_out, "w") as f:
                    f.write(self._render_sections(rows, "pipe") + "\n")
                ctx.logger.info(f"FaultToleranceReport written to {md_out}")
                # JSON sidecar — makes cross-test aggregation cheap.
                with open(json_out, "w") as f:
                    json.dump(
                        {"test_name": os.path.basename(ctx.log_dir), "rows": rows},
                        f,
                        indent=2,
                        default=str,
                    )
                ctx.logger.info(f"FaultToleranceReport (JSON) written to {json_out}")
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
        recovery_s: Optional[float] = None,
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
            "recovery_s": "N/A" if recovery_s is None else recovery_s,
            "success_before": succ_b,
            "failed_before": fail_b,
            "success_after": "N/A" if fault_ns is None else succ_a,
            "failed_after": "N/A" if fault_ns is None else fail_a,
            "latency_before_ms": _avg(lat_b),
            "latency_after_ms": ("N/A" if fault_ns is None else _avg(lat_a)),
        }

    @staticmethod
    def _render_sections(rows: list[dict[str, Any]], tablefmt: str = "pipe") -> str:
        """Render the report as three narrow sections.

        Stacking sections vertically keeps each table well within an 80-
        column terminal (and a slide column) instead of forcing horizontal
        scroll. Sections share the failure-name column so rows align by
        eye between them.
        """
        from tabulate import tabulate

        def _f(v: Any) -> str:
            return v if isinstance(v, str) else f"{v:.2f}"

        timing_headers = ["Failure", "Startup (s)", "Recovery (s)"]
        timing_body = [
            [r["failure"], _f(r["startup_s"]), _f(r["recovery_s"])] for r in rows
        ]

        counts_headers = [
            "Failure",
            "Success Before",
            "Failed Before",
            "Success After",
            "Failed After",
        ]
        counts_body = [
            [
                r["failure"],
                r["success_before"],
                r["failed_before"],
                r["success_after"],
                r["failed_after"],
            ]
            for r in rows
        ]

        latency_headers = ["Failure", "Latency Before (ms)", "Latency After (ms)"]
        latency_body = [
            [r["failure"], _f(r["latency_before_ms"]), _f(r["latency_after_ms"])]
            for r in rows
        ]

        sep = "\n\n"
        return sep.join(
            [
                "## Timing\n\n"
                + tabulate(timing_body, headers=timing_headers, tablefmt=tablefmt),
                "## Request counts (around fault)\n\n"
                + tabulate(counts_body, headers=counts_headers, tablefmt=tablefmt),
                "## Latency\n\n"
                + tabulate(latency_body, headers=latency_headers, tablefmt=tablefmt),
            ]
        )


# =============================================================================
# Error Breakdown Report
# =============================================================================


@dataclass
class ErrorBreakdownReport(Report):
    """Render aiperf's per-error-type breakdown for each load.

    Reads ``profile_export_aiperf.json`` (the aggregate aiperf summary)
    for each ``StartLoad`` event and emits one table per load showing
    each distinct error type, its cause chain, and the count. Useful
    for understanding *why* a fault scenario produced failures (request
    timeouts, server 5xx, connection resets, etc.).
    """

    save_to_file: bool = True

    def generate(self, ctx: "ScenarioContext") -> None:
        from tests.fault_tolerance.deploy.events import StartLoad

        load_events = [e for e in ctx.events if isinstance(e, StartLoad)]
        if not load_events:
            ctx.logger.info("ErrorBreakdownReport: no StartLoad in scenario; skipping")
            return

        sections: list[str] = []
        for load in load_events:
            ml = getattr(load, "_managed_load", None)
            local_dir = (
                getattr(ml, "local_output_dir", None) if ml else None
            ) or os.path.join(ctx.log_dir, "load")
            summary_path = os.path.join(local_dir, "profile_export_aiperf.json")
            if not os.path.exists(summary_path):
                ctx.logger.warning(f"ErrorBreakdownReport: {summary_path} not found")
                continue
            try:
                with open(summary_path) as f:
                    summary = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                ctx.logger.warning(f"ErrorBreakdownReport: load {load.name}: {e}")
                continue

            total_errors = (summary.get("error_request_count") or {}).get("avg", 0)
            entries = summary.get("error_summary") or []
            rows: list[list[Any]] = []
            for entry in entries:
                d = entry.get("error_details") or {}
                rows.append(
                    [
                        d.get("type", "unknown"),
                        d.get("message", ""),
                        " → ".join(d.get("cause_chain") or []),
                        entry.get("count", 0),
                    ]
                )
            if not rows:
                rows.append(["(none)", "—", "—", 0])

            section = self._render(load.name, int(total_errors), rows, "fancy_grid")
            ctx.logger.info("\n" + section)
            sections.append(self._render(load.name, int(total_errors), rows, "pipe"))

        if self.save_to_file and sections:
            out = os.path.join(ctx.log_dir, "error_breakdown_report.md")
            try:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w") as f:
                    f.write("\n\n".join(sections) + "\n")
                ctx.logger.info(f"ErrorBreakdownReport written to {out}")
            except OSError as e:
                ctx.logger.warning(f"ErrorBreakdownReport save failed: {e}")

    @property
    def description(self) -> str:
        return "Per-load aiperf error-type breakdown"

    @staticmethod
    def _render(
        load_name: str, total: int, rows: list[list[Any]], tablefmt: str
    ) -> str:
        from tabulate import tabulate

        headers = ["Type", "Message", "Cause Chain", "Count"]
        title = f"### Load '{load_name}' — total errors: {total}"
        return f"{title}\n\n{tabulate(rows, headers=headers, tablefmt=tablefmt)}"
