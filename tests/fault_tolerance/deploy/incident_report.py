# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Normalized incident-style report for fault_tolerance/deploy tests.

Produces a Markdown report with embedded PNG/HTML panels from a
``test_outputs/<test_name>/`` (or recovered) bundle directory:

  * **Throughput** — RPS + goodput per rung
  * **Latency p99** — TTFT + RL with 20 s SLA line
  * **Per-pod KV cache %** — one trace per pod, event markers overlaid
  * **Per-pod queue depth** — running + waiting per pod
  * **Errors per rung** — 503 vs TimeoutError stacked
  * **Pod restart timeline** — restart count over time
  * **Imbalance per rung** — peak gap / ratio / CV across per-pod metrics
  * **Gap-over-time** — `max − min` evolution for the worst-imbalance rung

Event timeline comes from parsing ``test.log.txt`` for ``Event N/M`` /
``DeletePod`` / ``WaitForRecovery`` / etc. — no new framework
instrumentation required.

Designed to work on both *complete* bundles (recovered long-A-data)
and *partial* bundles (kv-vs-ll-A where PVC extraction failed). Missing
panels produce an explicit "DATA INCOMPLETE" callout instead of zeroing.

Comparison mode: pass ``--compare-with <other_bundle>`` to render
two-trace overlays on every panel + side-by-side imbalance tables.

CLI:
    python -m tests.fault_tolerance.deploy.incident_report \\
        <bundle_dir> [--compare-with <other>] [--output-dir <dir>]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────
# Data model
# ───────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class EventMark:
    """A timestamped event from the scenario, parsed from test.log.txt."""

    ts: datetime  # absolute timestamp
    label: str  # short label, e.g. "Event 5/16: Start load 'c216'"
    raw: str  # full log line for context
    kind: str = "info"  # "info" | "rung_start" | "rung_end" | "fault" | "recovery"
    rung_name: Optional[str] = None
    severity: str = "info"  # affects rendering color
    detail: str = ""

    @property
    def color(self) -> str:
        return {
            "info": "rgba(120,120,120,0.6)",
            "rung_start": "rgba(30,144,255,0.7)",
            "rung_end": "rgba(30,144,255,0.5)",
            "fault": "rgba(220,20,60,0.85)",
            "recovery": "rgba(60,179,113,0.85)",
        }.get(self.kind, "rgba(120,120,120,0.6)")


@dataclasses.dataclass
class RungData:
    """Per-rung aggregate from profile_export_aiperf.json."""

    name: str  # e.g. "c72"
    concurrency: int
    dir_path: Path  # the load-c<N>-<hash>/ dir
    rps: float = 0.0
    goodput: float = 0.0
    ttft_p99_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_avg_ms: float = 0.0
    rl_p99_ms: float = 0.0
    rl_p50_ms: float = 0.0
    rl_avg_ms: float = 0.0
    itl_p50_ms: float = 0.0
    itl_p99_ms: float = 0.0
    valid_requests: int = 0
    error_503: int = 0
    error_timeout: int = 0
    has_aiperf: bool = False  # False = extraction failed for this rung

    @property
    def total_errors(self) -> int:
        return self.error_503 + self.error_timeout

    @property
    def total_requests(self) -> int:
        return self.valid_requests + self.total_errors


@dataclasses.dataclass
class PodInfo:
    """Pod identity from manifest YAML."""

    name: str
    ip: Optional[str]
    service: str  # "frontend" | "vllmprefillworker" | "vllmdecodeworker"
    yaml_path: Path

    @property
    def short(self) -> str:
        # Last 6 chars of pod name suffix — readable in plots
        return self.name.split("-")[-1][-6:]


@dataclasses.dataclass
class PodMetricSample:
    """One scrape sample of one metric on one pod."""

    ts: float  # epoch ns
    value: float


@dataclasses.dataclass
class Bundle:
    """A loaded test_outputs/<test_name>/ (or recovered) bundle directory."""

    root: Path
    label: str  # human label e.g. "kv-routing" / "least-loaded"
    test_name: str
    rungs: list[RungData] = dataclasses.field(default_factory=list)
    pods: list[PodInfo] = dataclasses.field(default_factory=list)
    events: list[EventMark] = dataclasses.field(default_factory=list)
    ip_to_svc: dict[str, str] = dataclasses.field(default_factory=dict)
    # raw per-rung per-url per-metric sample lists
    # samples[rung_name][svc][pod_short][metric_name] = [(ts_ns, value), ...]
    samples: dict = dataclasses.field(default_factory=dict)


# ───────────────────────────────────────────────────────────────────────
# Bundle loading
# ───────────────────────────────────────────────────────────────────────


def _maybe_first(path: Path, glob: str) -> Optional[Path]:
    matches = list(path.glob(glob))
    return matches[0] if matches else None


def _aiperf_get(j: dict, k: str, sub: str = "avg") -> float:
    v = j.get(k) or {}
    out = v.get(sub)
    if isinstance(out, (int, float)):
        return float(out)
    return 0.0


def _parse_aiperf_errors(aiperf_log: Path) -> tuple[int, int]:
    """Parse the final Error summary block — returns (503_count, timeout_count)."""
    if not aiperf_log.exists():
        return 0, 0
    try:
        content = aiperf_log.read_text(errors="ignore")
    except Exception:
        return 0, 0
    matches = re.findall(r"Error summary: \[(.*?)\] \(", content, re.DOTALL)
    if not matches:
        return 0, 0
    last = matches[-1]
    blocks = re.split(r"ErrorDetailsCount\(", last)
    per_type: dict[str, int] = {}
    for b in blocks[1:]:
        tm = re.search(r"type='([^']+)'", b)
        cm = re.search(r"count=(\d+)", b)
        if tm and cm:
            per_type[tm.group(1)] = int(cm.group(1))
    return per_type.get("Service Unavailable", 0), per_type.get("TimeoutError", 0)


def _parse_aiperf_valid(aiperf_log: Path) -> int:
    if not aiperf_log.exists():
        return 0
    try:
        for line in aiperf_log.open(errors="ignore"):
            m = re.search(r"Processed (\d+) valid requests", line)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return 0


def _load_rungs(bundle_root: Path) -> list[RungData]:
    """Find all `load-<name>-<hash>` dirs and parse aiperf JSON if present.

    Accepts BOTH the test_overload "c<N>" naming and arbitrary names like
    "heavy-kv" used by other scenarios.

    Handles BOTH layouts:
      bundle_root/load-<name>-<hash>/aiperf/profile_export_aiperf.json
      bundle_root/load/load-<name>-<hash>/profile_export_aiperf.json
    """
    rungs: list[RungData] = []
    candidates = list(bundle_root.glob("load/load-*-*")) + list(
        bundle_root.glob("load-*-*")
    )
    for d in sorted(candidates):
        m = re.match(r"load-(.+?)-[0-9a-f]{6,}$", d.name)
        if not m:
            continue
        tag = m.group(1)
        # `c<N>` rungs get the integer concurrency; arbitrary names get 0
        # (used only for sort ordering, not in display).
        cm = re.fullmatch(r"c(\d+)", tag)
        conc = int(cm.group(1)) if cm else 0
        # Try two possible layouts
        aiperf_dir = d / "aiperf" if (d / "aiperf").is_dir() else d
        pj = aiperf_dir / "profile_export_aiperf.json"
        log = aiperf_dir / "aiperf.log"

        rd = RungData(name=tag, concurrency=conc, dir_path=d)
        if pj.is_file():
            try:
                j = json.loads(pj.read_text())
            except Exception as e:
                logger.warning(f"Failed to parse {pj}: {e}")
                rungs.append(rd)
                continue
            rd.has_aiperf = True
            rd.rps = _aiperf_get(j, "request_throughput")
            rd.goodput = _aiperf_get(j, "goodput")
            rd.ttft_p99_ms = _aiperf_get(j, "time_to_first_token", "p99")
            rd.ttft_p50_ms = _aiperf_get(j, "time_to_first_token", "p50")
            rd.ttft_avg_ms = _aiperf_get(j, "time_to_first_token", "avg")
            rd.rl_p99_ms = _aiperf_get(j, "request_latency", "p99")
            rd.rl_p50_ms = _aiperf_get(j, "request_latency", "p50")
            rd.rl_avg_ms = _aiperf_get(j, "request_latency", "avg")
            rd.itl_p50_ms = _aiperf_get(j, "inter_token_latency", "p50")
            rd.itl_p99_ms = _aiperf_get(j, "inter_token_latency", "p99")
            rd.valid_requests = _parse_aiperf_valid(log)
            rd.error_503, rd.error_timeout = _parse_aiperf_errors(log)
        rungs.append(rd)
    rungs.sort(key=lambda r: r.concurrency)
    return rungs


def _load_pods(
    bundle_root: Path, extra_search_roots: Optional[list[Path]] = None
) -> tuple[list[PodInfo], dict[str, str]]:
    """Parse pod manifest YAMLs to build (PodInfo list, ip→service map).

    Searches under `bundle_root` AND any `extra_search_roots` for files
    matching {frontend,vllmprefillworker,vllmdecodeworker}/*.yaml. The
    recovered tarball lacks pod YAMLs (they live in the local
    `test_outputs/<test>/` bundle that the framework wrote at teardown);
    pass that as an extra root to merge."""
    pods: list[PodInfo] = []
    ip_to_svc: dict[str, str] = {}
    roots = [bundle_root] + (extra_search_roots or [])
    for root in roots:
        if not root.is_dir():
            continue
        for svc in ("frontend", "vllmprefillworker", "vllmdecodeworker"):
            for base in (root, root / "service_logs"):
                if not (base / svc).is_dir():
                    continue
                for f in (base / svc).glob("*.yaml"):
                    try:
                        y = yaml.safe_load(f.read_text())
                    except Exception:
                        continue
                    if not isinstance(y, dict) or y.get("kind") != "Pod":
                        continue
                    ip = (y.get("status") or {}).get("podIP")
                    name = (y.get("metadata") or {}).get("name", "?")
                    pods.append(PodInfo(name=name, ip=ip, service=svc, yaml_path=f))
                    if ip:
                        ip_to_svc[ip] = svc
    return pods, ip_to_svc


def _load_events(
    bundle_root: Path, extra_search_roots: Optional[list[Path]] = None
) -> list[EventMark]:
    """Extract timestamped events from test.log.txt by regex.

    The framework's log format is:
      2026-05-19 20:30:38 [INFO] test_decode_overload: Event 2/16: Start load 'c72'

    Searches under `bundle_root` first, then any `extra_search_roots`.
    """
    candidates = [bundle_root, bundle_root.parent] + (extra_search_roots or [])
    test_log = None
    for root in candidates:
        cand = root / "test.log.txt"
        if cand.is_file():
            test_log = cand
            break
    if test_log is None:
        return []
    events: list[EventMark] = []
    line_re = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[\w+\] [\w\[\]\-_.]+: (.*)"
    )
    interesting = re.compile(
        r"Event \d+/\d+:|DeletePod|TerminateProcess|StallProcess|WaitForRecovery"
        r"|Load .* completed|Load .* started|StopLoad|StartLoad:|Ready condition True"
    )
    for raw in test_log.read_text(errors="ignore").splitlines():
        m = line_re.match(raw)
        if not m:
            continue
        msg = m.group(2)
        if not interesting.search(msg):
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        ev = EventMark(ts=ts, label=msg[:80], raw=raw)
        # Classify
        if "DeletePod" in msg or "TerminateProcess" in msg or "StallProcess" in msg:
            ev.kind, ev.severity = "fault", "fault"
        elif "WaitForRecovery" in msg:
            ev.kind, ev.severity = "recovery", "info"
        elif "Start load" in msg:
            ev.kind = "rung_start"
            rm = re.search(r"Start load '(\w+)'", msg)
            if rm:
                ev.rung_name = rm.group(1)
        elif "Load '" in msg and "completed" in msg:
            ev.kind = "rung_end"
            rm = re.search(r"Load '(\w+)' completed", msg)
            if rm:
                ev.rung_name = rm.group(1)
        events.append(ev)
    return events


def _load_per_pod_samples(bundle: Bundle, metrics_of_interest: Iterable[str]) -> None:
    """Walk server_metrics_export.jsonl per rung, collect per-pod samples.

    Populates bundle.samples[rung_name][svc][pod_short][metric] = [(ts_ns, value), ...]
    """
    metrics_set = set(metrics_of_interest)
    for rd in bundle.rungs:
        aiperf_dir = (
            rd.dir_path / "aiperf" if (rd.dir_path / "aiperf").is_dir() else rd.dir_path
        )
        smfile = aiperf_dir / "server_metrics_export.jsonl"
        if not smfile.is_file():
            continue
        rung_bucket = bundle.samples.setdefault(rd.name, {})
        for line in smfile.open():
            try:
                r = json.loads(line)
            except Exception:
                continue
            url = r.get("endpoint_url", "")
            ipm = re.search(r"http://([0-9.]+):", url)
            if not ipm:
                continue
            ip = ipm.group(1)
            svc = bundle.ip_to_svc.get(ip)
            if svc is None:
                continue
            ts = r.get("timestamp_ns")
            if ts is None:
                continue
            metrics = r.get("metrics") or {}
            pod_short = ip[-6:].replace(".", "")
            svc_bucket = rung_bucket.setdefault(svc, {})
            pod_bucket = svc_bucket.setdefault(pod_short, {})
            for mname in metrics_set:
                v = metrics.get(mname)
                if not v:
                    continue
                # Standard vllm metric record is a list; take first entry's
                # value (for gauges) or sum (for counters).
                if isinstance(v, list) and v:
                    e = v[0]
                    if isinstance(e, dict):
                        val = e.get("value")
                        if val is None:
                            val = e.get("sum")
                        if isinstance(val, (int, float)):
                            pod_bucket.setdefault(mname, []).append((ts, float(val)))


# ───────────────────────────────────────────────────────────────────────
# Imbalance computation
# ───────────────────────────────────────────────────────────────────────


@dataclasses.dataclass
class ImbalanceRow:
    rung: str
    service: str
    metric: str
    pods: int
    pod_peaks: list[float]
    min_val: float
    max_val: float
    gap: float
    ratio: float
    cv: float

    @property
    def severity(self) -> str:
        if self.ratio >= 1.50:
            return "🟥"
        if self.ratio >= 1.20:
            return "🟨"
        return "🟩"


def compute_imbalance(bundle: Bundle) -> list[ImbalanceRow]:
    """For each (rung, service, metric_of_interest) compute per-pod peak +
    gap/ratio/CV across pods."""
    out: list[ImbalanceRow] = []
    # (service, metric) pairs we look at
    targets = [
        ("vllmdecodeworker", "vllm:kv_cache_usage_perc"),
        ("vllmdecodeworker", "vllm:num_requests_running"),
        ("vllmdecodeworker", "vllm:num_requests_waiting"),
        ("vllmprefillworker", "vllm:kv_cache_usage_perc"),
        ("vllmprefillworker", "vllm:nixl_bytes_transferred"),
    ]
    for rd in bundle.rungs:
        rung_samples = bundle.samples.get(rd.name, {})
        for svc, metric in targets:
            svc_samples = rung_samples.get(svc, {})
            peaks = [
                max(v for _, v in pod_samples[metric])
                for pod_samples in svc_samples.values()
                if pod_samples.get(metric)
            ]
            if len(peaks) < 2:
                continue
            mean = sum(peaks) / len(peaks)
            var = sum((p - mean) ** 2 for p in peaks) / len(peaks)
            stddev = var**0.5
            row = ImbalanceRow(
                rung=rd.name,
                service=svc,
                metric=metric,
                pods=len(peaks),
                pod_peaks=sorted(peaks, reverse=True),
                min_val=min(peaks),
                max_val=max(peaks),
                gap=max(peaks) - min(peaks),
                ratio=max(peaks) / max(min(peaks), 1e-9),
                cv=stddev / mean if mean > 0 else 0.0,
            )
            out.append(row)
    return out


# ───────────────────────────────────────────────────────────────────────
# Plot rendering
# ───────────────────────────────────────────────────────────────────────


# Fixed per-arm colors. First entry = primary bundle, second = compare_with.
# Same hue across throughput / latency / errors panels so a reader can tell
# at a glance which arm they're looking at.
_ARM_COLORS = ("#1f77b4", "#d62728")  # blue, crimson — high contrast


def _arm_color(idx: int) -> str:
    return _ARM_COLORS[idx % len(_ARM_COLORS)]


def _png(fig: go.Figure, path: Path) -> None:
    fig.write_html(path.with_suffix(".html"), include_plotlyjs="cdn")
    try:
        fig.write_image(path.with_suffix(".png"), width=1200, height=550, scale=2)
    except Exception as e:
        logger.warning(f"PNG export failed for {path.name}: {e}")


def _add_event_markers(fig: go.Figure, events: list[EventMark], t0: datetime) -> None:
    for ev in events:
        x = (ev.ts - t0).total_seconds()
        fig.add_vline(
            x=x,
            line=dict(color=ev.color, dash="dash", width=1),
            annotation_text=ev.label.split(":")[0][:30],
            annotation_position="top right",
            annotation_font=dict(size=9, color=ev.color),
        )


def render_throughput(
    bundle: Bundle, output_dir: Path, compare_with: Optional[Bundle] = None
) -> Path:
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Goodput RPS", "Total RPS (incl. errors)")
    )
    for idx, run in enumerate([bundle] + ([compare_with] if compare_with else [])):
        color = _arm_color(idx)
        x = [r.concurrency for r in run.rungs if r.has_aiperf]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[r.goodput for r in run.rungs if r.has_aiperf],
                mode="lines+markers",
                name=run.label,
                legendgroup=run.label,
                line=dict(width=3, color=color),
                marker=dict(size=10, color=color),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[r.rps for r in run.rungs if r.has_aiperf],
                mode="lines+markers",
                name=run.label,
                legendgroup=run.label,
                line=dict(width=3, color=color),
                marker=dict(size=10, color=color),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="concurrency", row=1, col=1)
    fig.update_xaxes(title_text="concurrency", row=1, col=2)
    fig.update_yaxes(title_text="RPS", row=1, col=1)
    fig.update_yaxes(title_text="RPS", row=1, col=2)
    fig.update_layout(
        title=f"Throughput — {bundle.label}"
        + (f" vs {compare_with.label}" if compare_with else ""),
        legend=dict(orientation="h", y=1.15),
        height=550,
    )
    p = output_dir / "throughput"
    _png(fig, p)
    return p


def render_latency(
    bundle: Bundle, output_dir: Path, compare_with: Optional[Bundle] = None
) -> Path:
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("TTFT p99 (s)", "Request latency p99 (s)")
    )
    for idx, run in enumerate([bundle] + ([compare_with] if compare_with else [])):
        color = _arm_color(idx)
        x = [r.concurrency for r in run.rungs if r.has_aiperf]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[r.ttft_p99_ms / 1000 for r in run.rungs if r.has_aiperf],
                mode="lines+markers",
                name=run.label,
                legendgroup=run.label,
                line=dict(width=3, color=color),
                marker=dict(size=10, color=color),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[r.rl_p99_ms / 1000 for r in run.rungs if r.has_aiperf],
                mode="lines+markers",
                name=run.label,
                legendgroup=run.label,
                line=dict(width=3, color=color),
                marker=dict(size=10, color=color),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    for col in (1, 2):
        fig.add_hline(
            y=20,
            line=dict(color="grey", dash="dash"),
            annotation_text="20s SLA",
            row=1,
            col=col,
        )
    fig.update_xaxes(title_text="concurrency", row=1, col=1)
    fig.update_xaxes(title_text="concurrency", row=1, col=2)
    fig.update_yaxes(title_text="seconds", row=1, col=1)
    fig.update_yaxes(title_text="seconds", row=1, col=2)
    fig.update_layout(
        title=f"Latency p99 — {bundle.label}"
        + (f" vs {compare_with.label}" if compare_with else ""),
        legend=dict(orientation="h", y=1.15),
        height=550,
    )
    p = output_dir / "latency_p99"
    _png(fig, p)
    return p


def render_errors(
    bundle: Bundle, output_dir: Path, compare_with: Optional[Bundle] = None
) -> Path:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Valid requests per rung", "Error rate per rung (%)"),
    )
    for idx, run in enumerate([bundle] + ([compare_with] if compare_with else [])):
        color = _arm_color(idx)
        x = [r.name for r in run.rungs if r.has_aiperf]
        valid = [r.valid_requests for r in run.rungs if r.has_aiperf]
        err_rate = [
            100.0 * r.total_errors / r.total_requests if r.total_requests else 0
            for r in run.rungs
            if r.has_aiperf
        ]
        fig.add_trace(
            go.Bar(
                x=x,
                y=valid,
                name=run.label,
                legendgroup=run.label,
                offsetgroup=run.label,
                marker_color=color,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=x,
                y=err_rate,
                name=run.label,
                legendgroup=run.label,
                offsetgroup=run.label,
                marker_color=color,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_xaxes(title_text="rung", row=1, col=1)
    fig.update_xaxes(title_text="rung", row=1, col=2)
    fig.update_yaxes(title_text="valid requests", row=1, col=1)
    fig.update_yaxes(title_text="error %", row=1, col=2, range=[0, 100])
    fig.update_layout(
        title=f"Errors — {bundle.label}"
        + (f" vs {compare_with.label}" if compare_with else ""),
        barmode="group",
        legend=dict(orientation="h", y=1.15),
        height=550,
    )
    p = output_dir / "errors"
    _png(fig, p)
    return p


def render_imbalance_timeseries(
    bundle: Bundle,
    output_dir: Path,
    metric: str,
    title: str,
    fname: str,
    ylabel: str,
    target_service: str = "vllmdecodeworker",
    mode: str = "absolute",
    compare_with: Optional[Bundle] = None,
) -> Path:
    """Single-metric imbalance timeseries — per rung, ONE line per arm.

    The line value is the "gap": `max - min` across all pods of
    `target_service` at each scrape timestamp. With `mode="pct"` the
    value is `(max - min) / max * 100` (percent delta of max).

    Replaces the per-pod overlay chart (which became unreadable with
    9+ pods × 2 arms). Single signal: how unbalanced is the pool over
    time?
    """
    runs = [bundle] + ([compare_with] if compare_with else [])
    rungs_present = [r.name for r in bundle.rungs if r.has_aiperf]
    if not rungs_present:
        fig = go.Figure()
        fig.add_annotation(text=f"No {metric} data", showarrow=False)
        p = output_dir / fname
        _png(fig, p)
        return p
    n = len(rungs_present)
    fig = make_subplots(rows=1, cols=n, subplot_titles=rungs_present, shared_yaxes=True)
    for col, rung_name in enumerate(rungs_present, start=1):
        for idx, run in enumerate(runs):
            color = _arm_color(idx)
            sm = run.samples.get(rung_name, {}).get(target_service, {})
            if not sm:
                continue
            # Gather all timestamped (ts, value) per pod
            pods_pts = []
            for pod_short, m_dict in sm.items():
                pts = m_dict.get(metric)
                if pts:
                    pods_pts.append(sorted(pts))
            if not pods_pts:
                continue
            # `max_abs` plots the worst pod's absolute value (useful for
            # signals like wait_for_remote_kvs where the level itself is
            # the cascade signature). Other modes (`absolute`, `pct`) need
            # at least 2 pods to compute imbalance.
            if mode != "max_abs" and len(pods_pts) < 2:
                continue
            # Compute gap-over-time: at each timestamp, take max-min across
            # pods (forward-fill latest known value per pod).
            all_ts = sorted({t for pts in pods_pts for t, _ in pts})
            if not all_ts:
                continue
            t0 = all_ts[0]
            gap_x, gap_y = [], []
            for ts in all_ts:
                latest = []
                for pts in pods_pts:
                    val = None
                    for t, v in pts:
                        if t <= ts:
                            val = v
                        else:
                            break
                    if val is not None:
                        latest.append(val)
                if not latest:
                    continue
                if mode != "max_abs" and len(latest) < 2:
                    continue
                hi = max(latest)
                lo = min(latest)
                if mode == "pct":
                    gap = ((hi - lo) / hi * 100.0) if hi > 1e-9 else 0.0
                elif mode == "max_abs":
                    gap = hi
                else:
                    gap = hi - lo
                gap_x.append((ts - t0) / 1e9)
                gap_y.append(gap)
            if not gap_x:
                continue
            fig.add_trace(
                go.Scatter(
                    x=gap_x,
                    y=gap_y,
                    mode="lines",
                    name=run.label,
                    legendgroup=run.label,
                    showlegend=(col == 1),
                    line=dict(width=3, color=color),
                ),
                row=1,
                col=col,
            )
    fig.update_xaxes(title_text="seconds in rung")
    fig.update_yaxes(title_text=ylabel, col=1)
    fig.update_layout(
        title=f"{title} — gap across {target_service} pool",
        legend=dict(orientation="h", y=-0.15),
        height=500,
    )
    p = output_dir / fname
    _png(fig, p)
    return p


def render_per_pod_metric(
    bundle: Bundle,
    output_dir: Path,
    metric: str,
    title: str,
    fname: str,
    ylabel: str,
    target_service: str = "vllmdecodeworker",
    compare_with: Optional[Bundle] = None,
) -> Path:
    """One per-pod time-series PER RUNG. Layout: one subplot per rung;
    A bundle traces solid, compare_with traces dashed."""
    runs = [bundle] + ([compare_with] if compare_with else [])
    # Decide subplot layout: one column per rung
    rungs_present = [r.name for r in bundle.rungs if r.has_aiperf]
    if not rungs_present:
        # Empty placeholder
        fig = go.Figure()
        fig.add_annotation(text=f"No {metric} data", showarrow=False)
        p = output_dir / fname
        _png(fig, p)
        return p
    n = len(rungs_present)
    fig = make_subplots(rows=1, cols=n, subplot_titles=rungs_present, shared_yaxes=True)
    for col, rung_name in enumerate(rungs_present, start=1):
        for run_idx, run in enumerate(runs):
            sm = run.samples.get(rung_name, {}).get(target_service, {})
            if not sm:
                continue
            t0 = None
            for pod_short, m_dict in sorted(sm.items()):
                pts = m_dict.get(metric)
                if not pts:
                    continue
                pts_sorted = sorted(pts)
                if t0 is None:
                    t0 = pts_sorted[0][0]
                x = [(t - t0) / 1e9 for t, _ in pts_sorted]
                y = [v for _, v in pts_sorted]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"{run.label}/{pod_short}" if col == 1 else None,
                        legendgroup=f"{run.label}/{pod_short}",
                        showlegend=(col == 1),
                        line=dict(dash="solid" if run_idx == 0 else "dash"),
                    ),
                    row=1,
                    col=col,
                )
    fig.update_xaxes(title_text="seconds in rung")
    fig.update_yaxes(title_text=ylabel, col=1)
    fig.update_layout(
        title=f"{title} — per-pod ({target_service})"
        + (
            f" — {bundle.label} solid, {compare_with.label} dashed"
            if compare_with
            else f" — {bundle.label}"
        ),
        legend=dict(orientation="h", y=-0.15),
        height=500,
    )
    p = output_dir / fname
    _png(fig, p)
    return p


def render_gap_timeseries(
    bundle: Bundle,
    output_dir: Path,
    rows: list[ImbalanceRow],
) -> Optional[Path]:
    """For the WORST-imbalance row in the bundle, plot per-pod time-series +
    max−min gap line."""
    if not rows:
        return None
    worst = max(rows, key=lambda r: r.ratio)
    sm = bundle.samples.get(worst.rung, {}).get(worst.service, {})
    if not sm:
        return None
    fig = go.Figure()
    pods_pts = []
    t0 = None
    for pod_short, m_dict in sorted(sm.items()):
        pts = m_dict.get(worst.metric)
        if not pts:
            continue
        pts_sorted = sorted(pts)
        if t0 is None:
            t0 = pts_sorted[0][0]
        pods_pts.append((pod_short, pts_sorted))
        fig.add_trace(
            go.Scatter(
                x=[(t - t0) / 1e9 for t, _ in pts_sorted],
                y=[v for _, v in pts_sorted],
                mode="lines",
                name=pod_short,
                opacity=0.6,
            )
        )
    # Compute gap series (re-bucket all timestamps; simple approach: at each
    # observation timestamp, take max − min across pods present at that time)
    if len(pods_pts) >= 2:
        # Aggregate all timestamps and interpolate / forward-fill
        all_ts = sorted({t for _, pts in pods_pts for t, _ in pts})
        gap_x, gap_y = [], []
        for ts in all_ts:
            vals = []
            for _, pts in pods_pts:
                # Find sample at or before ts
                latest = None
                for t, v in pts:
                    if t <= ts:
                        latest = v
                    else:
                        break
                if latest is not None:
                    vals.append(latest)
            if len(vals) >= 2:
                gap_x.append((ts - t0) / 1e9)
                gap_y.append(max(vals) - min(vals))
        fig.add_trace(
            go.Scatter(
                x=gap_x,
                y=gap_y,
                mode="lines",
                name="max−min gap",
                line=dict(color="red", width=3, dash="dot"),
            )
        )
    fig.update_layout(
        title=f"Worst-imbalance gap over time — {worst.service} {worst.metric} @ {worst.rung}",
        xaxis_title="seconds in rung",
        yaxis_title=worst.metric,
        height=500,
        legend=dict(orientation="h", y=-0.15),
    )
    p = output_dir / "imbalance_gap_worst_rung"
    _png(fig, p)
    return p


# ───────────────────────────────────────────────────────────────────────
# Markdown assembly
# ───────────────────────────────────────────────────────────────────────


def _md_imbalance_table(
    rows: list[ImbalanceRow], compare_rows: Optional[list[ImbalanceRow]] = None
) -> str:
    """Render imbalance table in Markdown. Highlights rows with ratio>=1.20."""
    if not rows:
        return (
            "_No per-pod metrics available (single-pod test or sample data missing)._\n"
        )
    lines = []
    lines.append(
        "| sev | rung | service | metric | pods | min | max | gap | ratio | CV |"
    )
    lines.append(
        "|----:|:-----|:--------|:-------|----:|----:|----:|----:|------:|----:|"
    )
    for r in rows:
        # Skip "obviously balanced" rows when we have many — keep the table readable
        if r.ratio < 1.10 and r.cv < 0.05:
            continue
        lines.append(
            f"| {r.severity} | {r.rung} | {r.service.replace('vllm','')} | "
            f"{r.metric} | {r.pods} | "
            f"{r.min_val:.3f} | {r.max_val:.3f} | "
            f"{r.gap:.3f} | {r.ratio:.2f} | {r.cv:.2f} |"
        )
    if len(lines) == 2:
        return "_All metrics within 10% balance ratio (well-balanced)._\n"
    return "\n".join(lines) + "\n"


def write_report(
    bundle: Bundle,
    output_dir: Path,
    compare_with: Optional[Bundle] = None,
) -> Path:
    """Compose Markdown + render all panels into output_dir/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "panels"
    plots_dir.mkdir(exist_ok=True)

    # Render panels
    p_thr = render_throughput(bundle, plots_dir, compare_with)
    p_lat = render_latency(bundle, plots_dir, compare_with)
    p_err = render_errors(bundle, plots_dir, compare_with)
    # Imbalance time-series — one line per arm per rung
    p_kv_gap = render_imbalance_timeseries(
        bundle,
        plots_dir,
        metric="vllm:kv_cache_usage_perc",
        title="Decode KV imbalance",
        fname="imbalance_decode_kv",
        ylabel="max − min (KV %)",
        target_service="vllmdecodeworker",
        mode="absolute",
        compare_with=compare_with,
    )
    p_run_gap = render_imbalance_timeseries(
        bundle,
        plots_dir,
        metric="vllm:num_requests_running",
        title="Decode running-queue imbalance",
        fname="imbalance_decode_running",
        ylabel="max − min (running)",
        target_service="vllmdecodeworker",
        mode="absolute",
        compare_with=compare_with,
    )
    p_wait_gap = render_imbalance_timeseries(
        bundle,
        plots_dir,
        metric="vllm:num_requests_waiting",
        title="Decode waiting-queue imbalance",
        fname="imbalance_decode_waiting",
        ylabel="max − min (waiting)",
        target_service="vllmdecodeworker",
        mode="absolute",
        compare_with=compare_with,
    )
    p_pf_kv_gap = render_imbalance_timeseries(
        bundle,
        plots_dir,
        metric="vllm:kv_cache_usage_perc",
        title="Prefill KV imbalance",
        fname="imbalance_prefill_kv",
        ylabel="max − min (KV %)",
        target_service="vllmprefillworker",
        mode="absolute",
        compare_with=compare_with,
    )
    # Cascade signature panels — absolute worst-pod levels, NOT imbalance.
    # `wait_for_remote_kvs` is the deadlock signal that climbs from
    # single digits into the thousands during a cascade. running /
    # waiting + inflight let us cross-check the engine's view against
    # the router's view.
    p_wait_remote = render_imbalance_timeseries(
        bundle,
        plots_dir,
        metric="derived:wait_for_remote_kvs",
        title="Decode WAITING_FOR_REMOTE_KVS (derived)",
        fname="cascade_wait_for_remote",
        ylabel="max across decode pods (requests)",
        target_service="vllmdecodeworker",
        mode="max_abs",
        compare_with=compare_with,
    )
    p_inflight = render_imbalance_timeseries(
        bundle,
        plots_dir,
        metric="dynamo_component_inflight_requests",
        title="Decode inflight (router view)",
        fname="cascade_inflight",
        ylabel="max across decode pods",
        target_service="vllmdecodeworker",
        mode="max_abs",
        compare_with=compare_with,
    )
    p_running = render_imbalance_timeseries(
        bundle,
        plots_dir,
        metric="vllm:num_requests_running",
        title="Decode running (engine view)",
        fname="cascade_running",
        ylabel="max across decode pods",
        target_service="vllmdecodeworker",
        mode="max_abs",
        compare_with=compare_with,
    )
    p_waiting = render_imbalance_timeseries(
        bundle,
        plots_dir,
        metric="vllm:num_requests_waiting",
        title="Decode waiting queue (engine view, excl. waiting-for-remote)",
        fname="cascade_waiting",
        ylabel="max across decode pods",
        target_service="vllmdecodeworker",
        mode="max_abs",
        compare_with=compare_with,
    )
    p_kv_abs = render_imbalance_timeseries(
        bundle,
        plots_dir,
        metric="vllm:kv_cache_usage_perc",
        title="Decode KV cache usage (worst pod)",
        fname="cascade_kv_usage",
        ylabel="max KV %",
        target_service="vllmdecodeworker",
        mode="max_abs",
        compare_with=compare_with,
    )

    # Imbalance computation
    imb_rows = compute_imbalance(bundle)
    cmp_rows = compute_imbalance(compare_with) if compare_with else None
    p_gap = render_gap_timeseries(bundle, plots_dir, imb_rows)

    # Compose Markdown
    md_path = output_dir / "incident_report.md"
    out: list[str] = []
    out.append(f"# Incident report — {bundle.test_name}")
    out.append("")
    out.append(f"**Bundle:** `{bundle.root}`")
    if compare_with:
        out.append(f"**Compared with:** `{compare_with.root}` ({compare_with.label})")
    out.append("")

    # Verdict summary
    n_present = sum(1 for r in bundle.rungs if r.has_aiperf)
    n_missing = sum(1 for r in bundle.rungs if not r.has_aiperf)
    out.append("## Headline summary")
    out.append("")
    out.append(f"- **Rungs with data:** {n_present}/{len(bundle.rungs)}")
    if n_missing > 0:
        out.append(
            f"- **⚠ Rungs MISSING aiperf data:** {n_missing} "
            "(PVC extraction failed — salvage manually)"
        )
    out.append(f"- **Pods captured (manifests):** {len(bundle.pods)}")
    out.append(f"- **Events parsed from test.log.txt:** {len(bundle.events)}")
    out.append("")

    # Per-rung quick table
    out.append("## Per-rung headline")
    out.append("")
    out.append(
        "| rung | RPS | goodput | TTFT p99 (s) | RL p99 (s) | valid | 503 | timeout |"
    )
    out.append(
        "|:----:|----:|--------:|-------------:|-----------:|------:|----:|--------:|"
    )
    for r in bundle.rungs:
        if r.has_aiperf:
            out.append(
                f"| {r.name} | {r.rps:.2f} | {r.goodput:.2f} | "
                f"{r.ttft_p99_ms/1000:.1f} | {r.rl_p99_ms/1000:.1f} | "
                f"{r.valid_requests} | {r.error_503} | {r.error_timeout} |"
            )
        else:
            out.append(f"| {r.name} | _no data_ | | | | | | |")
    out.append("")

    # Event timeline
    if bundle.events:
        out.append("## Event timeline (from test.log.txt)")
        out.append("")
        out.append("| time | kind | event |")
        out.append("|:-----|:-----|:------|")
        t0 = bundle.events[0].ts
        for ev in bundle.events:
            rel = (ev.ts - t0).total_seconds()
            out.append(f"| +{int(rel):>5}s | {ev.kind} | {ev.label} |")
        out.append("")

    # Panels
    out.append("## Throughput")
    out.append("")
    out.append(f"![throughput](panels/{p_thr.name}.png)")
    out.append("")
    out.append("## Latency p99")
    out.append("")
    out.append(f"![latency](panels/{p_lat.name}.png)")
    out.append("")
    out.append("## Errors per rung")
    out.append("")
    out.append(f"![errors](panels/{p_err.name}.png)")
    out.append("")
    out.append("## Decode KV imbalance — gap over time")
    out.append("")
    out.append(
        "Single line per arm: `max − min` of `vllm:kv_cache_usage_perc` "
        "across decode pods at each timestamp. A flat ~0 line means the "
        "decode pool is balanced; a wide / growing gap means one decode "
        "pod is hotter than peers."
    )
    out.append("")
    out.append(f"![decode kv gap](panels/{p_kv_gap.name}.png)")
    out.append("")
    out.append("## Decode running-queue imbalance — gap over time")
    out.append("")
    out.append(f"![decode running gap](panels/{p_run_gap.name}.png)")
    out.append("")
    out.append("## Decode waiting-queue imbalance — gap over time")
    out.append("")
    out.append(f"![decode waiting gap](panels/{p_wait_gap.name}.png)")
    out.append("")
    out.append("## Prefill KV imbalance — gap over time")
    out.append("")
    out.append(f"![prefill kv gap](panels/{p_pf_kv_gap.name}.png)")
    out.append("")

    # Cascade-signature panels — these surface the WAITING_FOR_REMOTE_KVS
    # deadlock signal that hides between the engine's `num_requests_running`
    # and the router's `dynamo_component_inflight_requests`.
    out.append("## Decode cascade signatures")
    out.append("")
    out.append(
        "Worst-pod absolute level (not imbalance). "
        "`derived:wait_for_remote_kvs = inflight − running − waiting` — "
        "the deadlock signal. During a KV-pressure cascade this climbs from "
        "single digits into the thousands, *before* `num_requests_running` "
        "falls and *before* KV pegs at 1.0. Inflight and running are shown "
        "alongside for cross-check (router-view vs engine-view)."
    )
    out.append("")
    out.append("### WAITING_FOR_REMOTE_KVS (derived)")
    out.append("")
    out.append(f"![wait for remote](panels/{p_wait_remote.name}.png)")
    out.append("")
    out.append("### Inflight (router view) vs Running (engine view) vs Waiting")
    out.append("")
    out.append(f"![inflight](panels/{p_inflight.name}.png)")
    out.append("")
    out.append(f"![running](panels/{p_running.name}.png)")
    out.append("")
    out.append(f"![waiting](panels/{p_waiting.name}.png)")
    out.append("")
    out.append("### Decode KV cache usage (worst pod, absolute)")
    out.append("")
    out.append(f"![kv usage](panels/{p_kv_abs.name}.png)")
    out.append("")

    # Imbalance
    out.append("## Pod imbalance")
    out.append("")
    out.append("Highlighting rows with `ratio >= 1.10` or `CV >= 0.05`.")
    out.append("Severity: 🟩 ratio<1.20 · 🟨 1.20-1.50 · 🟥 ratio>=1.50")
    out.append("")
    out.append("### " + bundle.label)
    out.append(_md_imbalance_table(imb_rows))
    if compare_with and cmp_rows is not None:
        out.append("### " + compare_with.label)
        out.append(_md_imbalance_table(cmp_rows))
    if p_gap:
        out.append("### Gap evolution — worst-imbalance rung in primary bundle")
        out.append("")
        out.append(f"![gap timeseries](panels/{p_gap.name}.png)")
        out.append("")

    md_path.write_text("\n".join(out))
    logger.info(f"wrote {md_path}")
    return md_path


# ───────────────────────────────────────────────────────────────────────
# Top-level entry
# ───────────────────────────────────────────────────────────────────────


def load_bundle(
    root: Path,
    label: Optional[str] = None,
    pod_manifests_from: Optional[Path] = None,
) -> Bundle:
    extras = [pod_manifests_from] if pod_manifests_from else []
    rungs = _load_rungs(root)
    pods, ip_to_svc = _load_pods(root, extra_search_roots=extras)
    events = _load_events(root, extra_search_roots=extras)
    test_name = root.name
    b = Bundle(
        root=root,
        label=label or root.name,
        test_name=test_name,
        rungs=rungs,
        pods=pods,
        events=events,
        ip_to_svc=ip_to_svc,
    )
    _load_per_pod_samples(
        b,
        metrics_of_interest={
            "vllm:kv_cache_usage_perc",
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:nixl_bytes_transferred",
            # `dynamo_component_inflight_requests` is the router-side
            # dispatched-not-completed count. Combined with running +
            # waiting on the engine side, the gap = WAITING_FOR_REMOTE_KVS
            # bucket — the deadlock signal that hides in stock metrics.
            "dynamo_component_inflight_requests",
        },
    )
    _derive_wait_for_remote(b)
    return b


def _derive_wait_for_remote(bundle: Bundle) -> None:
    """Synthesize `derived:wait_for_remote_kvs` per decode pod.

    Formula (per pod, per timestamp):
        wait_for_remote = dynamo_component_inflight_requests
                        - vllm:num_requests_running
                        - vllm:num_requests_waiting

    Joined on nearest-prior timestamp; written into the same samples
    dict so the existing render_imbalance_timeseries / render path
    just sees another decode metric.
    """
    for rung_name, svc_map in bundle.samples.items():
        decode = svc_map.get("vllmdecodeworker", {})
        for pod_short, metric_map in decode.items():
            inflight = metric_map.get("dynamo_component_inflight_requests") or []
            running = metric_map.get("vllm:num_requests_running") or []
            waiting = metric_map.get("vllm:num_requests_waiting") or []
            if not inflight or not running:
                continue

            # Build lookup dicts of (ts → val) for running/waiting; for
            # each inflight ts, take nearest-prior running+waiting.
            def _nearest_prior(series, ts):
                lo, hi = 0, len(series) - 1
                best = None
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if series[mid][0] <= ts:
                        best = series[mid][1]
                        lo = mid + 1
                    else:
                        hi = mid - 1
                return best

            running_sorted = sorted(running)
            waiting_sorted = sorted(waiting)
            derived = []
            for ts, inflight_v in sorted(inflight):
                run_v = _nearest_prior(running_sorted, ts)
                wait_v = _nearest_prior(waiting_sorted, ts) or 0
                if run_v is None:
                    continue
                gap = inflight_v - run_v - wait_v
                # Clip tiny negatives from sample skew (router can briefly
                # show inflight < running+waiting around request completion).
                derived.append((ts, max(0.0, gap)))
            if derived:
                metric_map["derived:wait_for_remote_kvs"] = derived


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "bundle", type=Path, help="Bundle dir (test_outputs/<test_name>/ or recovered)"
    )
    ap.add_argument(
        "--compare-with", type=Path, help="Second bundle for A/B comparison"
    )
    ap.add_argument("--label", default=None, help="Label for primary bundle")
    ap.add_argument("--compare-label", default=None, help="Label for comparison bundle")
    ap.add_argument(
        "--pod-manifests-from",
        type=Path,
        default=None,
        help="Extra search dir for pod YAMLs + test.log.txt "
        "(useful when running on recovered tarballs that lack manifests).",
    )
    ap.add_argument(
        "--compare-pod-manifests-from",
        type=Path,
        default=None,
        help="Extra search dir for comparison-bundle pod YAMLs + test.log.txt.",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: <bundle>/incident_report)",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    primary = load_bundle(
        args.bundle,
        label=args.label or args.bundle.name,
        pod_manifests_from=args.pod_manifests_from,
    )
    cmp_bundle = (
        load_bundle(
            args.compare_with,
            label=args.compare_label or args.compare_with.name,
            pod_manifests_from=args.compare_pod_manifests_from,
        )
        if args.compare_with
        else None
    )
    out_dir = args.output_dir or (args.bundle / "incident_report")
    md = write_report(primary, out_dir, compare_with=cmp_bundle)
    print(f"\nWrote incident report: {md}")


if __name__ == "__main__":
    main()
