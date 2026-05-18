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
                # Per-event timing block — every event in the scenario,
                # with its description + UTC wall-clock bracket. Lets
                # post-process tools (cascade_timeline --events, etc.)
                # overlay event boundaries on the per-second time-series
                # without needing to scrape pytest logs or kubectl jobs.
                events_block = []
                for evt in ctx.events:
                    started = getattr(evt, "started_at", None)
                    ended = getattr(evt, "ended_at", None)
                    events_block.append(
                        {
                            "type": type(evt).__name__,
                            "description": getattr(evt, "description", repr(evt)),
                            "name": getattr(evt, "name", None),
                            "started_at": started.isoformat() if started else None,
                            "ended_at": ended.isoformat() if ended else None,
                            "started_at_ns": (
                                int(started.timestamp() * 1e9) if started else None
                            ),
                            "ended_at_ns": (
                                int(ended.timestamp() * 1e9) if ended else None
                            ),
                        }
                    )
                # JSON sidecar — makes cross-test aggregation cheap.
                with open(json_out, "w") as f:
                    json.dump(
                        {
                            "test_name": os.path.basename(ctx.log_dir),
                            "rows": rows,
                            "events": events_block,
                        },
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


# =============================================================================
# Per-Worker Latency Report
# =============================================================================


@dataclass
class PerWorkerLatencyReport(Report):
    """Render per-worker latency-histogram p99 tables for each StartLoad.

    Reads ``server_metrics_export.jsonl`` (emitted by aiperf
    ``--server-metrics``) from each rung's local extract dir, computes
    the delta-histogram per pod across the rung's scrape window, and
    extracts p50 / p95 / p99 for the vLLM engine-side latency metrics.

    Lets readers attribute frontend TTFT spikes to specific workers and
    phases. For disagg in particular this separates:
      - prefill queue / prefill compute / NIXL post  (prefill role)
      - decode queue / decode TTFT / NIXL xfer / ITL (decode role)
    """

    save_to_file: bool = True

    # Metric -> column header. Order controls table column order.
    # NOTE: vllm:time_to_first_token_seconds on a *decode* worker in
    # disagg mode measures "engine arrival -> first decoded token",
    # which includes queue wait but excludes the NIXL transfer itself
    # (the request hits the engine queue only after NIXL completes).
    _METRICS = [
        ("vllm:request_queue_time_seconds", "queue"),
        ("vllm:time_to_first_token_seconds", "ttft"),
        ("vllm:request_prefill_time_seconds", "prefill"),
        ("vllm:request_decode_time_seconds", "decode"),
        ("vllm:request_inference_time_seconds", "inference"),
        ("vllm:nixl_xfer_time_seconds", "nixl_xfer"),
        ("vllm:nixl_post_time_seconds", "nixl_post"),
        ("vllm:inter_token_latency_seconds", "itl"),
        ("vllm:e2e_request_latency_seconds", "e2e"),
    ]

    def generate(self, ctx: "ScenarioContext") -> None:
        from tests.fault_tolerance.deploy.events import StartLoad

        load_events = [e for e in ctx.events if isinstance(e, StartLoad)]
        if not load_events:
            ctx.logger.info(
                "PerWorkerLatencyReport: no StartLoad in scenario; skipping"
            )
            return

        sections_screen: list[str] = []
        sections_file: list[str] = []
        for load in load_events:
            ml = getattr(load, "_managed_load", None)
            local_dir = (
                getattr(ml, "local_output_dir", None) if ml else None
            ) or os.path.join(ctx.log_dir, "load")
            jsonl = os.path.join(local_dir, "server_metrics_export.jsonl")
            if not os.path.exists(jsonl):
                ctx.logger.warning(
                    f"PerWorkerLatencyReport: {jsonl} not found for load "
                    f"{load.name!r}; aiperf may not have produced server "
                    "metrics. Pass concrete --server-metrics URLs in the "
                    "LoadConfig if running against a non-instrumented dgd."
                )
                continue

            by_pod = self._extract_per_worker_quantiles(jsonl)
            if not by_pod:
                ctx.logger.warning(
                    f"PerWorkerLatencyReport: no worker rows for {load.name!r}"
                )
                continue

            sections_screen.append(self._render(load.name, by_pod, "fancy_grid"))
            sections_file.append(self._render(load.name, by_pod, "pipe"))
            ctx.logger.info("\n" + sections_screen[-1])

        if self.save_to_file and sections_file:
            out = os.path.join(ctx.log_dir, "per_worker_latency_report.md")
            try:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w") as f:
                    f.write("\n\n".join(sections_file) + "\n")
                ctx.logger.info(f"PerWorkerLatencyReport written to {out}")
            except OSError as e:
                ctx.logger.warning(f"PerWorkerLatencyReport save failed: {e}")

    @property
    def description(self) -> str:
        return "Per-worker latency-histogram p99 (TTFT / queue / NIXL etc.)"

    @classmethod
    def _extract_per_worker_quantiles(
        cls, jsonl_path: str
    ) -> dict[str, dict[str, Any]]:
        """Walk the rung's server-metrics JSONL, compute per-worker p50/p95/p99.

        Returns ``{short_pod: {"role": str, metric_label: (p50, p95, p99), ...}}``.
        """
        first: dict[str, dict[str, dict]] = {}
        last: dict[str, dict[str, dict]] = {}
        role_by_ep: dict[str, str] = {}

        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ep = rec.get("endpoint_url", "")
                if "frontend" in ep:
                    continue
                metrics = rec.get("metrics", {})
                for k, _ in cls._METRICS:
                    series_list = metrics.get(k, [])
                    for series in series_list:
                        bk = series.get("buckets")
                        if not bk:
                            continue
                        comp = series.get("labels", {}).get("dynamo_component")
                        if comp and ep not in role_by_ep:
                            role_by_ep[ep] = comp
                        first.setdefault(ep, {}).setdefault(k, bk)
                        last.setdefault(ep, {})[k] = bk

        out: dict[str, dict[str, Any]] = {}
        for ep in sorted(last):
            row: dict[str, Any] = {"role": role_by_ep.get(ep, "?")}
            for k, label in cls._METRICS:
                f_bk = first.get(ep, {}).get(k)
                l_bk = last.get(ep, {}).get(k)
                if not f_bk or not l_bk:
                    row[label] = None
                    continue
                delta = {kk: max(0.0, l_bk.get(kk, 0) - f_bk.get(kk, 0)) for kk in l_bk}
                row[label] = (
                    cls._histo_quantile(delta, 0.50),
                    cls._histo_quantile(delta, 0.95),
                    cls._histo_quantile(delta, 0.99),
                )
            out[cls._short_pod(ep)] = row
        return out

    @staticmethod
    def _short_pod(ep: str) -> str:
        # http://100.67.X.Y:9090/metrics -> 100.67.X.Y
        return ep.replace("http://", "").split(":")[0]

    @staticmethod
    def _histo_quantile(delta: dict[str, float], q: float):
        """Inverse-CDF on a Prom-style cumulative histogram delta.

        Returns ``float`` (seconds) on success, the string ``">{le}s"``
        when the quantile falls in the +Inf bucket (no upper bound),
        or ``None`` if there's no data.
        """
        total = delta.get("+Inf", 0)
        items = sorted(
            ((float(k), v) for k, v in delta.items() if k != "+Inf"),
            key=lambda x: x[0],
        )
        if total <= 0 or not items:
            return None
        target = q * total
        prev_le, prev_count = 0.0, 0.0
        for le, c in items:
            if c >= target:
                if c == prev_count:
                    return le
                return prev_le + (target - prev_count) / (c - prev_count) * (
                    le - prev_le
                )
            prev_le, prev_count = le, c
        # In the +Inf bucket — no upper bound resolvable from histogram.
        return f">{items[-1][0]:.1f}"

    @classmethod
    def _render(
        cls,
        load_name: str,
        by_pod: dict[str, dict[str, Any]],
        tablefmt: str,
    ) -> str:
        from tabulate import tabulate

        labels = [label for _, label in cls._METRICS]

        # Prefill rows first, then decode/backend, then anything else;
        # within a role, sort lexicographically by short pod id.
        def role_sort_key(p):
            r = by_pod[p]["role"]
            order = {"prefill": 0, "decode": 1, "backend": 1}.get(r, 9)
            return (order, p)

        pods = sorted(by_pod, key=role_sort_key)
        headers = ["pod", "role"] + [f"{lbl}_p99" for lbl in labels]
        rows = []
        for p in pods:
            row = [p, by_pod[p]["role"]]
            for lbl in labels:
                val = by_pod[p].get(lbl)
                row.append(cls._fmt_val(val))
            rows.append(row)
        title = f"### Per-worker latency p99 — load '{load_name}'"
        return f"{title}\n\n{tabulate(rows, headers=headers, tablefmt=tablefmt)}"

    @staticmethod
    def _fmt_val(val):
        if val is None:
            return "-"
        # val is (p50, p95, p99) -- show p99 (table title says p99).
        p99 = val[2]
        if p99 is None:
            return "-"
        if isinstance(p99, str):
            return p99
        if p99 >= 1.0:
            return f"{p99:.2f}s"
        return f"{p99 * 1000:.0f}ms"


# =============================================================================
# GPU Memory Report
# =============================================================================


@dataclass
class GpuMemoryReport(Report):
    """Per-rung max GPU framebuffer usage from DCGM_FI_DEV_FB_USED.

    Queries Prometheus over each StartLoad rung's wall-clock window for
    every GPU the deployment's pods used (via DCGM's
    `exported_namespace` / `exported_pod` labels). Emits a table of
    max GB-used per GPU per rung and flags anything > `max_gb_per_gpu`.

    Used to verify the A100-40GB emulation: with
    `--gpu-memory-utilization=0.45` on H100-80GB, vLLM should fit
    within ~36 GB plus a few-hundred-MB of driver / NIXL overhead;
    anything > 40 GB means the cap isn't being honored.
    """

    save_to_file: bool = True
    # Soft threshold — exceeding this is logged loudly but not asserted.
    # Hard enforcement should be a separate Check (so failures don't
    # block report emission).
    max_gb_per_gpu: float = 40.0
    prometheus_url: str = "http://localhost:9090"
    # Match the namespace the deployment runs in. Read from the
    # ScenarioContext at run time.
    _namespace: Optional[str] = None
    # Match pod-name substring; auto-populated from the deployment.
    _dgd_name: Optional[str] = None

    def generate(self, ctx: "ScenarioContext") -> None:
        from tests.fault_tolerance.deploy.events import StartLoad

        # Get pod name substring from the deployment spec.
        ns = ctx.namespace
        dgd_name = getattr(getattr(ctx, "deployment", None), "_deployment_name", None)
        if not dgd_name and getattr(ctx, "deployment", None) is not None:
            dgd_name = ctx.deployment.deployment_spec.name
        # Common DGD-name pattern: pods are <dgd>-N-... ; match by substring.

        load_events = [e for e in ctx.events if isinstance(e, StartLoad)]
        if not load_events:
            ctx.logger.info("GpuMemoryReport: no StartLoad in scenario; skipping")
            return

        rows: list[dict] = []
        for load in load_events:
            started = getattr(load, "started_at", None)
            ended = getattr(load, "ended_at", None)
            if not started or not ended:
                continue
            per_gpu = self._query_max_fb_used(ns, dgd_name, started, ended)
            for label, mb in per_gpu.items():
                rows.append({"rung": load.name, "gpu": label, "max_gb": mb / 1024.0})

        if not rows:
            ctx.logger.warning(
                "GpuMemoryReport: no DCGM samples returned. Check Prometheus URL "
                "(%s) and that DCGM_FI_DEV_FB_USED has labels exported_namespace=%s.",
                self.prometheus_url,
                ns,
            )
            return

        # Render two views: per-(rung, pod, gpu) raw table + per-rung max.
        screen = self._render(rows, "fancy_grid")
        ctx.logger.info("\n" + screen)
        if self.save_to_file:
            out = os.path.join(ctx.log_dir, "gpu_memory_report.md")
            try:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w") as f:
                    f.write(self._render(rows, "pipe") + "\n")
                ctx.logger.info(f"GpuMemoryReport written to {out}")
            except OSError as e:
                ctx.logger.warning(f"GpuMemoryReport save failed: {e}")

        # Loud-log any GPU exceeding the threshold.
        violators = [r for r in rows if r["max_gb"] > self.max_gb_per_gpu]
        if violators:
            for v in violators:
                ctx.logger.error(
                    "GpuMemoryReport: %s on rung %s used %.2f GB (> %.2f GB cap)",
                    v["gpu"],
                    v["rung"],
                    v["max_gb"],
                    self.max_gb_per_gpu,
                )
        else:
            ctx.logger.info(
                "GpuMemoryReport: all GPUs stayed within %.2f GB across all rungs.",
                self.max_gb_per_gpu,
            )

    @property
    def description(self) -> str:
        return "Per-rung max GPU framebuffer usage (DCGM)"

    def _query_max_fb_used(self, ns, dgd_name, started, ended):
        """Return ``{<short_label>: max_MB}`` for every GPU used by the
        DGD's pods during ``[started, ended]``."""
        import urllib.parse
        import urllib.request

        win_s = max(1, int((ended - started).total_seconds()))
        end_ts = int(ended.timestamp())
        # DCGM exports `exported_namespace` and `exported_pod` labels so
        # we can scope by deployment without touching the dgxc-alloy /
        # gpu-operator scrape config.
        selector = f'exported_namespace="{ns}"'
        if dgd_name:
            selector += f',exported_pod=~".*{dgd_name}.*"'
        query = f"max_over_time(DCGM_FI_DEV_FB_USED{{{selector}}}[{win_s}s])"
        url = f"{self.prometheus_url}/api/v1/query?" + urllib.parse.urlencode(
            {"query": query, "time": end_ts}
        )
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                d = json.loads(r.read())
        except Exception:
            return {}
        out: dict[str, float] = {}
        for s in d.get("data", {}).get("result", []):
            m = s["metric"]
            pod = m.get("exported_pod") or "?"
            gpu = m.get("gpu") or m.get("device") or "?"
            # Pod tail + gpu id keeps labels short and unique
            short = f"{pod.rsplit('-', 1)[-1]}/gpu{gpu}"
            try:
                v = float(s["value"][1])
            except (TypeError, ValueError):
                continue
            if v > out.get(short, 0):
                out[short] = v
        return out

    def _render(self, rows: list[dict], tablefmt: str) -> str:
        from tabulate import tabulate

        # Pivot: rows of rung × gpu showing max_gb
        rungs = sorted({r["rung"] for r in rows}, key=self._rung_sort_key)
        gpus = sorted({r["gpu"] for r in rows})
        cell = {(r["rung"], r["gpu"]): r["max_gb"] for r in rows}
        headers = ["rung"] + gpus
        table = []
        for rung in rungs:
            row = [rung]
            for g in gpus:
                v = cell.get((rung, g))
                row.append(f"{v:.2f}" if v is not None else "-")
            table.append(row)
        title = (
            f"### GPU framebuffer max GB used per rung "
            f"(DCGM_FI_DEV_FB_USED; cap = {self.max_gb_per_gpu:.1f} GB)"
        )
        return f"{title}\n\n{tabulate(table, headers=headers, tablefmt=tablefmt)}"

    @staticmethod
    def _rung_sort_key(name: str):
        # c64, r16 etc. — sort by phase letter then numeric suffix.
        if not name:
            return (9, 0, name)
        phase = name[0]
        try:
            n = int(name[1:])
        except ValueError:
            n = 0
        order = {"c": 0, "r": 1}.get(phase, 9)
        return (order, n, name)


# =============================================================================
# Error Tracking Report
# =============================================================================


@dataclass
class ErrorTrackingReport(Report):
    """One row per rung, aggregating every kind of error we can collect.

    Sources:
      - aiperf:           `error_request_count` + `error_summary` (from
                          profile_export_aiperf.json on each rung's local dir)
      - vLLM workers:     delta of `vllm:num_preemptions_total` /
                          `vllm:nixl_num_failed_transfers` /
                          `vllm:nixl_num_kv_expired_reqs` between the rung's
                          first and last server_metrics_export.jsonl scrape
      - k8s containers:   `increase(kube_pod_container_status_restarts_total[rung_window])`
                          summed across DGD pods; plus the most recent
                          `kube_pod_container_status_last_terminated_reason`
                          (OOMKilled, Error, Completed, ...) so we know
                          WHY a restart happened. kube-state-metrics
                          retains the series even after pods are gone,
                          so this works at report time too.
    """

    save_to_file: bool = True
    prometheus_url: str = "http://localhost:9090"

    def generate(self, ctx: "ScenarioContext") -> None:
        from tests.fault_tolerance.deploy.events import StartLoad

        load_events = [e for e in ctx.events if isinstance(e, StartLoad)]
        if not load_events:
            ctx.logger.info("ErrorTrackingReport: no StartLoad in scenario; skipping")
            return

        ns = ctx.namespace
        dgd_name = None
        if getattr(ctx, "deployment", None) is not None:
            dgd_name = ctx.deployment.deployment_spec.name

        rows = []
        for load in load_events:
            ml = getattr(load, "_managed_load", None)
            local_dir = (
                getattr(ml, "local_output_dir", None) if ml else None
            ) or os.path.join(ctx.log_dir, "load")
            row = {"rung": load.name}

            # aiperf error count + breakdown
            ai_total, ai_types = self._aiperf_errors(
                os.path.join(local_dir, "profile_export_aiperf.json")
            )
            row["aiperf_errors"] = ai_total
            row["aiperf_error_types"] = ai_types

            # vLLM worker error counters (sum across pods of per-pod delta).
            row.update(
                self._vllm_error_deltas(
                    os.path.join(local_dir, "server_metrics_export.jsonl")
                )
            )

            # Container restart count + reason during the rung window.
            started = getattr(load, "started_at", None)
            ended = getattr(load, "ended_at", None)
            restarts, reasons = self._container_restarts(ns, dgd_name, started, ended)
            row["container_restarts"] = restarts
            row["restart_reasons"] = reasons

            rows.append(row)

        # Append a per-pod log-file-count summary. The dyn_tee.sh
        # wrapper suffixes each container start's log with the start
        # time (`<pod>_<ts>.log`), so the count of log files per pod
        # equals the number of container starts. >1 file means the
        # container restarted at least once.
        log_summary = self._log_file_summary(ctx.log_dir)

        screen = self._render(rows, "fancy_grid") + "\n\n" + log_summary
        ctx.logger.info("\n" + screen)
        if self.save_to_file:
            out = os.path.join(ctx.log_dir, "error_tracking_report.md")
            try:
                os.makedirs(os.path.dirname(out), exist_ok=True)
                with open(out, "w") as f:
                    f.write(self._render(rows, "pipe") + "\n\n" + log_summary + "\n")
                ctx.logger.info(f"ErrorTrackingReport written to {out}")
            except OSError as e:
                ctx.logger.warning(f"ErrorTrackingReport save failed: {e}")

    @property
    def description(self) -> str:
        return (
            "Per-rung errors: aiperf error count + vLLM preemption / NIXL "
            "failure / KV-expired counters"
        )

    @staticmethod
    def _aiperf_errors(path: str):
        if not os.path.exists(path):
            return 0, ""
        try:
            with open(path) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            return 0, ""
        total = int((d.get("error_request_count") or {}).get("avg", 0))
        # Compact summary: "<count> <type>" comma-joined
        bits = []
        for entry in d.get("error_summary") or []:
            t = (entry.get("error_details") or {}).get("type", "?")
            bits.append(f"{entry.get('count', 0)} {t}")
        return total, ", ".join(bits)

    @staticmethod
    def _vllm_error_deltas(jsonl_path: str):
        """Per-rung sum-across-pods delta of vLLM error counters.

        Returns dict with keys:
          preemptions, nixl_failed_xfers, nixl_kv_expired
        """
        out = {"preemptions": 0, "nixl_failed_xfers": 0, "nixl_kv_expired": 0}
        if not os.path.exists(jsonl_path):
            return out
        counters = {
            "vllm:num_preemptions_total": "preemptions",
            "vllm:nixl_num_failed_transfers": "nixl_failed_xfers",
            "vllm:nixl_num_kv_expired_reqs": "nixl_kv_expired",
        }
        first: dict[str, dict[str, float]] = {}
        last: dict[str, dict[str, float]] = {}
        try:
            with open(jsonl_path) as f:
                for line in f:
                    rec = json.loads(line)
                    ep = rec.get("endpoint_url", "")
                    if "frontend" in ep:
                        continue
                    for k, _label in counters.items():
                        for s in rec.get("metrics", {}).get(k, []):
                            v = s.get("value")
                            if v is None:
                                continue
                            first.setdefault(ep, {}).setdefault(k, float(v))
                            last.setdefault(ep, {})[k] = float(v)
        except (json.JSONDecodeError, OSError):
            return out
        for ep in last:
            f_per = first.get(ep, {})
            l_per = last.get(ep, {})
            for k, label in counters.items():
                delta = l_per.get(k, 0) - f_per.get(k, 0)
                if delta > 0:
                    out[label] += int(delta)
        return out

    @staticmethod
    def _log_file_summary(log_dir: str) -> str:
        """Walk the extracted service-logs tree and count log files per pod.

        With the dyn_tee.sh wrapper writing one log file per container
        start (`<pod-name>_<unix-ts>.log`), a pod that restarted has
        more than one file. Returns a markdown bullet list.
        """
        if not os.path.isdir(log_dir):
            return "_(no log dir found for restart-artifact summary)_"

        # Each component subdir (frontend / vllmprefillworker /
        # vllmdecodeworker) contains <pod>_<ts>.log. Strip the
        # `_<digits>.log` suffix to group log files by pod name.
        import re
        from collections import defaultdict

        TS_SUFFIX = re.compile(r"_(\d+)\.log$")
        per_pod: dict[str, list[str]] = defaultdict(list)
        for root, _dirs, files in os.walk(log_dir):
            for fn in files:
                m = TS_SUFFIX.search(fn)
                if not m:
                    continue
                pod = TS_SUFFIX.sub("", fn)
                per_pod[pod].append(fn)

        restarted = [(p, len(fs)) for p, fs in per_pod.items() if len(fs) > 1]
        lines = [
            "### Container-restart log-file summary",
            "",
            f"Total pods with log files: {len(per_pod)}",
            f"Pods with > 1 log file (= restarted): {len(restarted)}",
        ]
        if restarted:
            lines.extend(
                [
                    "",
                    "| pod | log files (= container starts) | restarts |",
                    "|---|---:|---:|",
                ]
            )
            for p, count in sorted(restarted):
                lines.append(f"| {p} | {count} | {count - 1} |")
        else:
            lines.append("")
            lines.append("_All containers ran to completion without restart._")
        return "\n".join(lines)

    def _container_restarts(self, ns, dgd_name, started, ended):
        """Query Prometheus (kube-state-metrics) for restart count +
        most-recent-terminated reasons during ``[started, ended]``.

        Returns (count: int, reasons_summary: str).
        """
        if started is None or ended is None or not ns:
            return 0, ""
        import urllib.parse
        import urllib.request

        win_s = max(1, int((ended - started).total_seconds()))
        end_ts = int(ended.timestamp())
        selector = f'namespace="{ns}"'
        if dgd_name:
            selector += f',pod=~".*{dgd_name}.*"'

        def _q(query):
            url = f"{self.prometheus_url}/api/v1/query?" + urllib.parse.urlencode(
                {"query": query, "time": end_ts}
            )
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    return json.loads(r.read())
            except Exception:
                return {}

        # Sum increase over the rung window across all matching pods.
        d = _q(
            f"sum(increase(kube_pod_container_status_restarts_total{{{selector}}}[{win_s}s]))"
        )
        try:
            total = int(round(float(d["data"]["result"][0]["value"][1])))
        except (KeyError, IndexError, ValueError, TypeError):
            total = 0

        if total == 0:
            return 0, ""

        # Group per-pod with reason. We use last_terminated_reason as the
        # most informative signal — its label `reason` carries
        # OOMKilled / Error / Completed / etc.
        d = _q(f"kube_pod_container_status_last_terminated_reason{{{selector}}} == 1")
        reasons = {}
        for s in d.get("data", {}).get("result", []):
            m = s["metric"]
            pod = m.get("pod", "?")
            reason = m.get("reason", "?")
            reasons[pod] = reason

        bits = sorted(f"{pod.rsplit('-', 1)[-1]}:{r}" for pod, r in reasons.items())
        return total, ", ".join(bits)

    @staticmethod
    def _render(rows: list[dict], tablefmt: str) -> str:
        from tabulate import tabulate

        headers = [
            "rung",
            "aiperf_errors",
            "aiperf_error_types",
            "preemptions",
            "nixl_failed_xfers",
            "nixl_kv_expired",
            "container_restarts",
            "restart_reasons",
        ]
        table = []
        for r in rows:
            table.append(
                [
                    r.get("rung", "?"),
                    r.get("aiperf_errors", 0),
                    r.get("aiperf_error_types", ""),
                    r.get("preemptions", 0),
                    r.get("nixl_failed_xfers", 0),
                    r.get("nixl_kv_expired", 0),
                    r.get("container_restarts", 0),
                    r.get("restart_reasons", ""),
                ]
            )
        title = "### Error tracking — per-rung aiperf + vLLM error counters + container restarts"
        return f"{title}\n\n{tabulate(table, headers=headers, tablefmt=tablefmt)}"
