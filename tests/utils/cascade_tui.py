# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
cascade_tui — local TUI for live observation of a Dynamo DGD's per-worker metrics.

Standalone: needs only kubectl + a kubeconfig. No Grafana required. Scrapes each
worker pod's :9090/metrics directly via kubectl port-forward, parses with aiperf's
ServerMetricsDataCollector, and renders a textual app showing per-worker:

    pod | component | dp_rank | running | waiting | kv_pct | nixl_p99 | preempt

with 60-sample sparklines underneath each numeric column.

Tails an event log file (default `/tmp/cascade_events.log`) for fault-injection
markers from the cascade_inject CLI; events show up as bottom-pane log lines and
mark the next sample on the sparkline.

Press `q` or Ctrl-C to quit. Press `r/w/k/n` to swap the sparkline focus metric
(running/waiting/kv/nixl).

Usage (from dev container with /root/.kube mounted):

    KUBECONFIG=/root/.kube/config python -m tests.utils.cascade_tui \\
        --context nv-prd-dgxc.teleport.sh-dynamo-aws-dev-02 \\
        --namespace neelays-test \\
        --dgd vllm-disagg-qwen3-30b-2p1d
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("cascade_tui")

# textual imports gated so --help works even on hosts without textual.
try:
    from textual.app import App, ComposeResult
    from textual.widgets import DataTable, Footer, Header, Log

    HAVE_TEXTUAL = True
except ImportError:
    HAVE_TEXTUAL = False

try:
    from aiperf.common.models import ErrorDetails  # noqa: F401
    from aiperf.server_metrics.data_collector import (  # noqa: F401
        ServerMetricsDataCollector,
    )

    HAVE_AIPERF = True
except ImportError:
    HAVE_AIPERF = False


# Histogram quantile estimator — same as scripts/spike_view.py
def histogram_quantile(buckets, q: float) -> Optional[float]:
    if not buckets:
        return None
    ordered = []
    for k, v in buckets.items() if isinstance(buckets, dict) else buckets:
        try:
            le = float(k) if k not in ("+Inf", "Inf") else float("inf")
        except (ValueError, TypeError):
            continue
        ordered.append((le, float(v)))
    ordered.sort()
    if not ordered or ordered[-1][1] == 0:
        return None
    total = ordered[-1][1]
    target = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in ordered:
        if count >= target:
            if le == float("inf"):
                return prev_le
            if count == prev_count:
                return le
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_count = le, count
    return ordered[-1][0]


# ────────────────────────────────────────────────────────────────────────────
# Pod discovery + port-forward helpers


@dataclass
class Worker:
    """A single Dynamo worker pod we're scraping."""

    pod_name: str
    pod_ip: str
    component: str  # "frontend" / "prefill" / "decode" (from labels)
    subcomponent: str  # operator's subComponentType
    local_port: int = 0  # set when port-forward is established
    pf_proc: Optional[subprocess.Popen] = None
    collector: Optional[object] = None
    # Latest scraped values
    running: float = 0.0
    waiting: float = 0.0
    kv_pct: float = 0.0
    nixl_p50: float = 0.0
    nixl_p99: float = 0.0
    preempt_total: float = 0.0
    ttft_p50: float = 0.0
    ttft_p99: float = 0.0
    itl_p50: float = 0.0
    itl_p99: float = 0.0
    e2e_p50: float = 0.0
    e2e_p99: float = 0.0
    rps: float = 0.0
    mem_mib: float = 0.0
    last_update: float = 0.0
    # Counter-tracking for RPS (delta / time)
    _last_req_count: float = 0.0
    _last_req_t: float = 0.0
    # Rolling history for sparklines (1 sample/sec, ~60s)
    history_running: deque = field(default_factory=lambda: deque(maxlen=60))
    history_waiting: deque = field(default_factory=lambda: deque(maxlen=60))
    history_kv: deque = field(default_factory=lambda: deque(maxlen=60))
    history_nixl: deque = field(default_factory=lambda: deque(maxlen=60))
    history_ttft: deque = field(default_factory=lambda: deque(maxlen=60))
    history_itl: deque = field(default_factory=lambda: deque(maxlen=60))
    history_e2e: deque = field(default_factory=lambda: deque(maxlen=60))
    history_rps: deque = field(default_factory=lambda: deque(maxlen=60))
    history_mem: deque = field(default_factory=lambda: deque(maxlen=60))


async def discover_workers(
    namespace: str, dgd_name: str, kube_context: Optional[str]
) -> list[Worker]:
    """List pods matching the DGD label and produce Worker descriptors."""
    from kubernetes_asyncio import client, config

    if kube_context:
        await config.load_kube_config(context=kube_context)
    else:
        await config.load_kube_config()

    v1 = client.CoreV1Api()
    label_selector = f"nvidia.com/dynamo-graph-deployment-name={dgd_name}"
    pods = await v1.list_namespaced_pod(
        namespace=namespace, label_selector=label_selector
    )
    workers = []
    for p in pods.items:
        if p.status.phase != "Running":
            continue
        labels = p.metadata.labels or {}
        comp = labels.get("nvidia.com/dynamo-component", "?")
        sub = labels.get("nvidia.com/dynamo-subcomponent", "")
        workers.append(
            Worker(
                pod_name=p.metadata.name,
                pod_ip=p.status.pod_ip or "",
                component=comp,
                subcomponent=sub,
            )
        )
    await v1.api_client.close()
    return workers


def _free_port() -> int:
    """Reserve a random free local port. (Not race-safe but good enough.)"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _remote_metrics_port(worker: Worker) -> int:
    """Frontend pods expose /metrics on :8000 (OpenAI port shared with
    dynamo_frontend_* gauges); worker pods (Prefill/Decode) publish
    vllm:* metrics on :9090."""
    if worker.component == "Frontend":
        return 8000
    return 9090


def start_port_forward(
    worker: Worker, namespace: str, kube_context: Optional[str]
) -> None:
    """Spawn `kubectl port-forward` from a free local port to pod's
    metrics port (8000 for Frontend, 9090 for workers)."""
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl not on PATH — needed for port-forward")
    worker.local_port = _free_port()
    remote_port = _remote_metrics_port(worker)
    cmd = ["kubectl"]
    if kube_context:
        cmd += ["--context", kube_context]
    cmd += [
        "-n",
        namespace,
        "port-forward",
        f"pod/{worker.pod_name}",
        f"{worker.local_port}:{remote_port}",
    ]
    # Discard kubectl's stdout/stderr — chatty; we observe via scraping.
    worker.pf_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


def stop_port_forward(worker: Worker) -> None:
    if worker.pf_proc and worker.pf_proc.poll() is None:
        worker.pf_proc.terminate()
        try:
            worker.pf_proc.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            worker.pf_proc.kill()


# ────────────────────────────────────────────────────────────────────────────
# Metric extraction from a ServerMetricsRecord


METRIC_RUNNING = "vllm:num_requests_running"
METRIC_WAITING = "vllm:num_requests_waiting"
METRIC_KV = "vllm:kv_cache_usage_perc"
METRIC_NIXL_HIST = "vllm:nixl_post_time_seconds"
METRIC_PREEMPT = "vllm:num_preemptions_total"


def update_worker_from_record(worker: Worker, record) -> None:
    """Pull our 5 metrics out of an aiperf ServerMetricsRecord and update worker state."""
    metrics = getattr(record, "metrics", None)
    if not metrics:
        return

    def first_value(family_name: str) -> Optional[float]:
        family = metrics.get(family_name)
        if not family or not getattr(family, "samples", None):
            return None
        # If multiple samples (e.g. labeled per dp_rank), aggregate by sum
        total = 0.0
        seen = False
        for s in family.samples:
            if getattr(s, "value", None) is not None:
                total += float(s.value)
                seen = True
        return total if seen else None

    def first_histogram_p99(family_name: str) -> Optional[float]:
        family = metrics.get(family_name)
        if not family or not getattr(family, "samples", None):
            return None
        # Aggregate over all samples — sum buckets
        agg_buckets: dict[float, float] = {}
        for s in family.samples:
            buckets = getattr(s, "buckets", None) or {}
            if isinstance(buckets, dict):
                items = buckets.items()
            else:
                items = buckets
            for le, cnt in items:
                try:
                    le_f = float(le) if str(le) not in ("+Inf", "Inf") else float("inf")
                except (ValueError, TypeError):
                    continue
                agg_buckets[le_f] = agg_buckets.get(le_f, 0.0) + float(cnt)
        if not agg_buckets:
            return None
        return histogram_quantile(agg_buckets, 0.99)

    r = first_value(METRIC_RUNNING)
    if r is not None:
        worker.running = r
        worker.history_running.append(r)
    w = first_value(METRIC_WAITING)
    if w is not None:
        worker.waiting = w
        worker.history_waiting.append(w)
    k = first_value(METRIC_KV)
    if k is not None:
        worker.kv_pct = k
        worker.history_kv.append(k)
    n = first_histogram_p99(METRIC_NIXL_HIST)
    if n is not None:
        worker.nixl_p99 = n
        worker.history_nixl.append(n)
    p = first_value(METRIC_PREEMPT)
    if p is not None:
        worker.preempt_total = p
    worker.last_update = time.time()


# ────────────────────────────────────────────────────────────────────────────
# Direct HTTP scrape — replaces aiperf ServerMetricsDataCollector
# Robust against aiperf API drift; parses Prometheus exposition format inline.


def _parse_prom_text(text: str) -> dict:
    """Minimal Prometheus exposition-format parser.

    Returns ``{metric_name: {(label_tuple): value}}`` for gauges/counters,
    plus ``{metric_name + "_bucket": {(label_tuple_without_le, le): count}}``
    for histograms. Skips HELP/TYPE comments and malformed lines.

    Far simpler than pulling in `prometheus_client` — we only need a handful
    of metric families and don't care about distinguishing gauge from counter.
    """
    out: dict = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        # name{labels} value [timestamp]
        if "{" in line:
            name, rest = line.split("{", 1)
            labels_str, val_str = rest.split("}", 1)
            val_str = val_str.strip().split()[0]
            try:
                val = float(val_str)
            except ValueError:
                continue
            labels = {}
            for kv in labels_str.split(","):
                if "=" not in kv:
                    continue
                k, v = kv.split("=", 1)
                labels[k.strip()] = v.strip().strip('"')
            out.setdefault(name, []).append((labels, val))
        else:
            try:
                name, val_str = line.split(maxsplit=1)
                val = float(val_str.split()[0])
            except ValueError:
                continue
            out.setdefault(name, []).append(({}, val))
    return out


def _sum_gauge(samples: list, **filter_labels) -> Optional[float]:
    """Sum values across all samples matching the optional label filters."""
    if not samples:
        return None
    total = 0.0
    seen = False
    for labels, val in samples:
        if any(labels.get(k) != v for k, v in filter_labels.items()):
            continue
        total += val
        seen = True
    return total if seen else None


def _histogram_p99(buckets_samples: list) -> Optional[float]:
    """Compute p99 from Prometheus histogram bucket samples (le-labeled).

    Aggregates across all label sets by summing bucket counts per `le`.
    """
    if not buckets_samples:
        return None
    agg: dict[float, float] = {}
    for labels, val in buckets_samples:
        le = labels.get("le")
        if le is None:
            continue
        try:
            le_f = float("inf") if le in ("+Inf", "Inf") else float(le)
        except ValueError:
            continue
        agg[le_f] = agg.get(le_f, 0.0) + val
    if not agg:
        return None
    return histogram_quantile(agg, 0.99)


def _histogram_quantiles(buckets_samples: list, *quantiles) -> dict:
    """Compute multiple quantiles from one aggregated bucket-set.

    Avoids re-walking the per-label samples for each quantile. Returns
    ``{q: value}`` for each requested q (default p50, p99 if none given).
    """
    if not buckets_samples:
        return {}
    if not quantiles:
        quantiles = (0.5, 0.99)
    agg: dict[float, float] = {}
    for labels, val in buckets_samples:
        le = labels.get("le")
        if le is None:
            continue
        try:
            le_f = float("inf") if le in ("+Inf", "Inf") else float(le)
        except ValueError:
            continue
        agg[le_f] = agg.get(le_f, 0.0) + val
    if not agg:
        return {}
    return {q: histogram_quantile(agg, q) for q in quantiles}


async def scrape_worker_metrics_http(worker: Worker, session) -> None:
    """Fetch /metrics from worker.local_port and update the Worker state.

    Handles both worker pods (vllm:* metrics on :9090) and Frontend pods
    (dynamo_frontend_* + dynamo_component_router_* on :8000). Uses aiohttp
    for non-blocking I/O so the textual app stays responsive.
    """
    import aiohttp

    url = f"http://localhost:{worker.local_port}/metrics"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
            text = await resp.text()
    except Exception:
        return  # transient — try again next tick
    parsed = _parse_prom_text(text)

    now = time.time()
    if worker.component == "Frontend":
        # FE columns — running=active, waiting=disconnects, nixl_p99=N/A
        # (FE has no NIXL but column header reserved for worker-mode).
        active = _sum_gauge(parsed.get("dynamo_frontend_active_requests", []))
        if active is not None:
            worker.running = active
            worker.history_running.append(active)
        disc = _sum_gauge(parsed.get("dynamo_frontend_disconnected_clients", []))
        if disc is not None:
            worker.waiting = disc
            worker.history_waiting.append(disc)
        ttft_q = _histogram_quantiles(
            parsed.get("dynamo_component_router_time_to_first_token_seconds_bucket", [])
        )
        if 0.5 in ttft_q:
            worker.ttft_p50 = ttft_q[0.5] or 0.0
        if 0.99 in ttft_q:
            worker.ttft_p99 = ttft_q[0.99] or 0.0
            worker.history_ttft.append(worker.ttft_p99)
        itl_q = _histogram_quantiles(
            parsed.get("dynamo_component_router_inter_token_latency_seconds_bucket", [])
        )
        if 0.5 in itl_q:
            worker.itl_p50 = itl_q[0.5] or 0.0
        if 0.99 in itl_q:
            worker.itl_p99 = itl_q[0.99] or 0.0
            worker.history_itl.append(worker.itl_p99)
        e2e_q = _histogram_quantiles(
            parsed.get("dynamo_frontend_request_duration_seconds_bucket", [])
        )
        if 0.5 in e2e_q:
            worker.e2e_p50 = e2e_q[0.5] or 0.0
        if 0.99 in e2e_q:
            worker.e2e_p99 = e2e_q[0.99] or 0.0
            worker.history_e2e.append(worker.e2e_p99)
        # RPS: delta of dynamo_frontend_request_duration_seconds_count / dt
        req_count = _sum_gauge(
            parsed.get("dynamo_frontend_request_duration_seconds_count", [])
        )
        if req_count is not None:
            if worker._last_req_t > 0:
                dt = now - worker._last_req_t
                worker.rps = (
                    max(0.0, (req_count - worker._last_req_count) / dt)
                    if dt > 0
                    else 0.0
                )
                worker.history_rps.append(worker.rps)
            worker._last_req_count = req_count
            worker._last_req_t = now
        stalls = _sum_gauge(parsed.get("dynamo_frontend_event_loop_stall_total", []))
        if stalls is not None:
            worker.preempt_total = stalls
    else:
        # Worker pods — vllm:* gauges + histograms.
        r = _sum_gauge(parsed.get(METRIC_RUNNING, []))
        if r is not None:
            worker.running = r
            worker.history_running.append(r)
        w = _sum_gauge(parsed.get(METRIC_WAITING, []))
        if w is not None:
            worker.waiting = w
            worker.history_waiting.append(w)
        k = _sum_gauge(parsed.get(METRIC_KV, []))
        if k is not None:
            worker.kv_pct = k
            worker.history_kv.append(k)
        nixl_q = _histogram_quantiles(parsed.get(f"{METRIC_NIXL_HIST}_bucket", []))
        if 0.5 in nixl_q:
            worker.nixl_p50 = nixl_q[0.5] or 0.0
        if 0.99 in nixl_q:
            worker.nixl_p99 = nixl_q[0.99] or 0.0
            worker.history_nixl.append(worker.nixl_p99)
        ttft_q = _histogram_quantiles(
            parsed.get("vllm:time_to_first_token_seconds_bucket", [])
        )
        if 0.5 in ttft_q:
            worker.ttft_p50 = ttft_q[0.5] or 0.0
        if 0.99 in ttft_q:
            worker.ttft_p99 = ttft_q[0.99] or 0.0
            worker.history_ttft.append(worker.ttft_p99)
        itl_q = _histogram_quantiles(
            parsed.get("vllm:inter_token_latency_seconds_bucket", [])
        )
        if 0.5 in itl_q:
            worker.itl_p50 = itl_q[0.5] or 0.0
        if 0.99 in itl_q:
            worker.itl_p99 = itl_q[0.99] or 0.0
            worker.history_itl.append(worker.itl_p99)
        e2e_q = _histogram_quantiles(
            parsed.get("vllm:e2e_request_latency_seconds_bucket", [])
        )
        if 0.5 in e2e_q:
            worker.e2e_p50 = e2e_q[0.5] or 0.0
        if 0.99 in e2e_q:
            worker.e2e_p99 = e2e_q[0.99] or 0.0
            worker.history_e2e.append(worker.e2e_p99)
        # RPS from vllm:request_success_total (sum over finished_reason labels)
        req_count = _sum_gauge(parsed.get("vllm:request_success_total", []))
        if req_count is not None:
            if worker._last_req_t > 0:
                dt = now - worker._last_req_t
                worker.rps = (
                    max(0.0, (req_count - worker._last_req_count) / dt)
                    if dt > 0
                    else 0.0
                )
                worker.history_rps.append(worker.rps)
            worker._last_req_count = req_count
            worker._last_req_t = now
        p = _sum_gauge(parsed.get(METRIC_PREEMPT, []))
        if p is not None:
            worker.preempt_total = p
    worker.last_update = now


# ────────────────────────────────────────────────────────────────────────────
# Sparkline rendering for the table cells

SPARK_CHARS = "▁▂▃▄▅▆▇█"


def render_spark(values, width: int = 16) -> str:
    if not values:
        return " " * width
    clean = list(values)
    if not clean:
        return " " * width
    lo, hi = min(clean), max(clean)
    if lo == hi:
        return SPARK_CHARS[3] * min(len(clean), width)
    span = hi - lo
    n = len(clean)
    if n <= width:
        binned = clean
    else:
        # Take the latest `width` samples (deque is already capped at maxlen=60 elsewhere)
        binned = list(clean)[-width:]
    out = []
    for v in binned:
        idx = int((v - lo) / span * (len(SPARK_CHARS) - 1))
        out.append(SPARK_CHARS[max(0, min(idx, len(SPARK_CHARS) - 1))])
    return "".join(out)


# ────────────────────────────────────────────────────────────────────────────
# Textual app


if HAVE_TEXTUAL:

    class CascadeTUI(App):
        """Live per-worker metrics + event log."""

        CSS = """
        Screen { layout: vertical; }
        #workers-table { height: 1fr; }
        #event-log { height: 10; }
        """
        BINDINGS = [
            ("q", "quit", "Quit"),
            ("ctrl+c", "quit", "Quit"),
        ]

        def __init__(
            self,
            namespace: str,
            dgd: str,
            kube_context: Optional[str],
            event_log_path: Optional[Path],
        ):
            super().__init__()
            self.namespace = namespace
            self.dgd = dgd
            self.kube_context = kube_context
            self.event_log_path = event_log_path
            self.dgd_workers: list[Worker] = []
            self._poll_task: Optional[asyncio.Task] = None
            self._evtlog_task: Optional[asyncio.Task] = None
            self._evtlog_pos = 0

        def compose(self) -> ComposeResult:
            yield Header()
            yield DataTable(id="workers-table", zebra_stripes=True, cursor_type="row")
            yield Log(id="event-log", highlight=True)
            yield Footer()

        async def on_mount(self) -> None:
            tbl = self.query_one("#workers-table", DataTable)
            # Explicit column keys so update_cell(row_key, col_key, ...)
            # in _refresh_row resolves cleanly. Without `key=` textual
            # uses internal generated keys and update_cell silently fails.
            # Slim columns to fit a wider set. p99-latency in ms.
            tbl.add_column("pod", width=42, key="pod")
            tbl.add_column("comp", width=8, key="comp")
            tbl.add_column("run", width=14, key="running")
            tbl.add_column("wait", width=14, key="waiting")
            tbl.add_column("kv%", width=14, key="kv%")
            tbl.add_column("rps", width=10, key="rps")
            tbl.add_column("ttft 50/99 ms", width=18, key="ttft")
            tbl.add_column("itl 50/99 ms", width=18, key="itl")
            tbl.add_column("e2e 50/99 ms", width=18, key="e2e")
            tbl.add_column("nixl 50/99 ms", width=18, key="nixl")
            tbl.add_column("mem MiB", width=10, key="mem MiB")
            tbl.add_column("preempts", width=8, key="preempts")

            evt = self.query_one("#event-log", Log)
            evt.write_line(
                f"discovering pods for DGD={self.dgd} in {self.namespace} ..."
            )
            try:
                self.dgd_workers = await discover_workers(
                    self.namespace, self.dgd, self.kube_context
                )
            except Exception as e:
                evt.write_line(f"ERROR discovering pods: {e}")
                self.dgd_workers = []

            evt.write_line(f"found {len(self.dgd_workers)} running pod(s)")
            for w in self.dgd_workers:
                tbl.add_row(
                    w.pod_name,
                    w.component,
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    key=w.pod_name,
                )

            # Establish port-forwards. Different remote port per component:
            # Frontend → :8000 (where dynamo_frontend_* + router metrics
            # are served), workers → :9090 (vllm:* gauges).
            for w in self.dgd_workers:
                try:
                    start_port_forward(w, self.namespace, self.kube_context)
                    evt.write_line(
                        f"port-forward → {w.pod_name} ({w.component}) "
                        f"localhost:{w.local_port} → :{_remote_metrics_port(w)}"
                    )
                except Exception as e:
                    evt.write_line(f"port-forward failed for {w.pod_name}: {e}")

            # Give port-forwards a moment to come up before scraping starts
            await asyncio.sleep(2.0)
            self._poll_task = asyncio.create_task(self._poll_loop())
            if self.event_log_path:
                self._evtlog_task = asyncio.create_task(self._tail_event_log())

        async def _poll_loop(self) -> None:
            """1 Hz direct HTTP scrape of every worker via aiohttp.

            Replaces the (broken under aiperf 0.8) ServerMetricsDataCollector
            path with a direct Prometheus-text fetch + parser. Same Worker
            state objects updated; table refresh tick stays at 1 Hz.
            """
            import aiohttp

            evt = self.query_one("#event-log", Log)
            tbl = self.query_one("#workers-table", DataTable)

            session = aiohttp.ClientSession()

            async def poll_memory():
                """Poll kubectl top every 10s for per-pod RSS."""
                while True:
                    try:
                        cmd = [
                            "kubectl",
                            "-n",
                            self.namespace,
                            "top",
                            "pod",
                            "--no-headers",
                        ]
                        if self.kube_context:
                            cmd[:0] = []  # context lives in kubeconfig already
                        proc = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.DEVNULL,
                        )
                        out, _ = await proc.communicate()
                        for line in out.decode().splitlines():
                            parts = line.split()
                            if len(parts) < 3:
                                continue
                            pod_name, _, mem_str = parts[0], parts[1], parts[2]
                            for w in self.dgd_workers:
                                if w.pod_name != pod_name:
                                    continue
                                # mem_str like "1234Mi" or "1Gi"
                                try:
                                    if mem_str.endswith("Mi"):
                                        m = float(mem_str[:-2])
                                    elif mem_str.endswith("Gi"):
                                        m = float(mem_str[:-2]) * 1024
                                    else:
                                        m = float(mem_str) / 1048576.0
                                    w.mem_mib = m
                                    w.history_mem.append(m)
                                except ValueError:
                                    pass
                                break
                    except Exception:
                        pass
                    await asyncio.sleep(10.0)

            mem_task = asyncio.create_task(poll_memory())
            try:
                fail_count: dict[str, int] = {}
                while True:
                    # Fan-out concurrent scrapes; gather lets one slow pod
                    # not block the others.
                    tasks = []
                    for w in self.dgd_workers:
                        if not w.local_port:
                            continue
                        tasks.append(scrape_worker_metrics_http(w, session))
                    if tasks:
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                        for w, res in zip(self.dgd_workers, results):
                            if isinstance(res, Exception):
                                key = w.pod_name
                                fail_count[key] = fail_count.get(key, 0) + 1
                                if fail_count[key] == 5:
                                    evt.write_line(
                                        f"scrape error [{w.pod_name}]: {res} "
                                        "(suppressing further reports)"
                                    )
                    for w in self.dgd_workers:
                        self._refresh_row(tbl, w)
                    await asyncio.sleep(1.0)
            finally:
                mem_task.cancel()
                await session.close()

        def _refresh_row(self, tbl, w: Worker) -> None:
            spark_w = 8
            running_cell = (
                f"{w.running:>4.0f} {render_spark(w.history_running, spark_w)}"
            )
            waiting_cell = (
                f"{w.waiting:>4.0f} {render_spark(w.history_waiting, spark_w)}"
            )
            kv_cell = f"{w.kv_pct:>5.1%} {render_spark(w.history_kv, spark_w)}"
            rps_cell = f"{w.rps:>5.1f}"

            def _fmt_pair(p50, p99, hist):
                p50_ms = p50 * 1000.0 if p50 else 0.0
                p99_ms = p99 * 1000.0 if p99 else 0.0
                return f"{p50_ms:>4.0f}/{p99_ms:>4.0f} {render_spark(hist, 6)}"

            ttft_cell = _fmt_pair(w.ttft_p50, w.ttft_p99, w.history_ttft)
            itl_cell = _fmt_pair(w.itl_p50, w.itl_p99, w.history_itl)
            e2e_cell = _fmt_pair(w.e2e_p50, w.e2e_p99, w.history_e2e)
            nixl_cell = _fmt_pair(w.nixl_p50, w.nixl_p99, w.history_nixl)
            mem_cell = f"{w.mem_mib:>5.0f}"
            preempt_cell = f"{int(w.preempt_total)}"
            try:
                tbl.update_cell(w.pod_name, "running", running_cell)
                tbl.update_cell(w.pod_name, "waiting", waiting_cell)
                tbl.update_cell(w.pod_name, "kv%", kv_cell)
                tbl.update_cell(w.pod_name, "rps", rps_cell)
                tbl.update_cell(w.pod_name, "ttft", ttft_cell)
                tbl.update_cell(w.pod_name, "itl", itl_cell)
                tbl.update_cell(w.pod_name, "e2e", e2e_cell)
                tbl.update_cell(w.pod_name, "nixl", nixl_cell)
                tbl.update_cell(w.pod_name, "mem MiB", mem_cell)
                tbl.update_cell(w.pod_name, "preempts", preempt_cell)
            except Exception as e:
                # Surface the first failure per pod to the event log so we
                # can debug; subsequent ones swallowed to avoid spam.
                key = ("_refresh_err", w.pod_name)
                if not getattr(self, "_seen_refresh_err", set()):
                    self._seen_refresh_err = set()
                if key not in self._seen_refresh_err:
                    self._seen_refresh_err.add(key)
                    try:
                        evt = self.query_one("#event-log", Log)
                        evt.write_line(f"refresh err [{w.pod_name}]: {e}")
                    except Exception:
                        pass

        async def _tail_event_log(self) -> None:
            """Tail the cascade-inject event log file and stream lines into the Log widget."""
            evt_widget = self.query_one("#event-log", Log)
            path = self.event_log_path
            assert path is not None
            # Wait for the file to exist
            while not path.exists():
                await asyncio.sleep(1.0)
            self._evtlog_pos = path.stat().st_size  # only show new lines from now
            while True:
                try:
                    new_size = path.stat().st_size
                    if new_size > self._evtlog_pos:
                        with path.open() as f:
                            f.seek(self._evtlog_pos)
                            chunk = f.read()
                            self._evtlog_pos = f.tell()
                        for line in chunk.splitlines():
                            if line.strip():
                                evt_widget.write_line(f"[event] {line}")
                    elif new_size < self._evtlog_pos:
                        # File rotated/truncated
                        self._evtlog_pos = 0
                except FileNotFoundError:
                    pass
                await asyncio.sleep(1.0)

        async def on_unmount(self) -> None:
            if self._poll_task:
                self._poll_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._poll_task
            for w in self.dgd_workers:
                if w.collector:
                    with contextlib.suppress(Exception):
                        await w.collector.stop()
                stop_port_forward(w)


# ────────────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--namespace", required=True)
    ap.add_argument("--dgd", required=True, help="DynamoGraphDeployment name")
    ap.add_argument(
        "--context",
        default=os.environ.get("KUBE_CONTEXT") or None,
        help="Kubeconfig context (default: current)",
    )
    ap.add_argument(
        "--event-log",
        type=Path,
        default=Path("/tmp/cascade_events.log"),
        help="Path to event log file written by cascade-inject CLI",
    )
    ap.add_argument(
        "--no-event-log",
        action="store_true",
        help="Don't tail an event log",
    )
    args = ap.parse_args()

    if not HAVE_TEXTUAL:
        print("error: textual not installed; pip install textual", file=sys.stderr)
        return 2
    if not HAVE_AIPERF:
        print(
            "error: aiperf not importable; pip install aiperf>=0.7.0",
            file=sys.stderr,
        )
        return 2

    logging.basicConfig(level=logging.WARNING)
    app = CascadeTUI(
        namespace=args.namespace,
        dgd=args.dgd,
        kube_context=args.context,
        event_log_path=None if args.no_event_log else args.event_log,
    )
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
