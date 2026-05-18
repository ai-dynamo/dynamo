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
    from aiperf.common.models import ErrorDetails
    from aiperf.server_metrics.data_collector import ServerMetricsDataCollector

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
    nixl_p99: float = 0.0
    preempt_total: float = 0.0
    last_update: float = 0.0
    # Rolling history for sparklines (1 sample/sec, ~60s)
    history_running: deque = field(default_factory=lambda: deque(maxlen=60))
    history_waiting: deque = field(default_factory=lambda: deque(maxlen=60))
    history_kv: deque = field(default_factory=lambda: deque(maxlen=60))
    history_nixl: deque = field(default_factory=lambda: deque(maxlen=60))


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


def start_port_forward(
    worker: Worker, namespace: str, kube_context: Optional[str]
) -> None:
    """Spawn `kubectl port-forward` from a free local port to pod's :9090."""
    if not shutil.which("kubectl"):
        raise RuntimeError("kubectl not on PATH — needed for port-forward")
    worker.local_port = _free_port()
    cmd = ["kubectl"]
    if kube_context:
        cmd += ["--context", kube_context]
    cmd += [
        "-n",
        namespace,
        "port-forward",
        f"pod/{worker.pod_name}",
        f"{worker.local_port}:9090",
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
            self.workers: list[Worker] = []
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
            tbl.add_column("pod", width=42)
            tbl.add_column("comp", width=10)
            tbl.add_column("running", width=24)
            tbl.add_column("waiting", width=24)
            tbl.add_column("kv%", width=24)
            tbl.add_column("nixl_p99 (ms)", width=24)
            tbl.add_column("preempts", width=10)

            evt = self.query_one("#event-log", Log)
            evt.write_line(
                f"discovering pods for DGD={self.dgd} in {self.namespace} ..."
            )
            try:
                self.workers = await discover_workers(
                    self.namespace, self.dgd, self.kube_context
                )
            except Exception as e:
                evt.write_line(f"ERROR discovering pods: {e}")
                self.workers = []

            evt.write_line(f"found {len(self.workers)} running pod(s)")
            for w in self.workers:
                tbl.add_row(
                    w.pod_name,
                    w.component,
                    "—",
                    "—",
                    "—",
                    "—",
                    "—",
                    key=w.pod_name,
                )

            # Establish port-forwards
            for w in self.workers:
                try:
                    start_port_forward(w, self.namespace, self.kube_context)
                    evt.write_line(
                        f"port-forward → {w.pod_name} ({w.component}) "
                        f"localhost:{w.local_port} → :9090"
                    )
                except Exception as e:
                    evt.write_line(f"port-forward failed for {w.pod_name}: {e}")

            # Give port-forwards a moment to come up before scraping starts
            await asyncio.sleep(2.0)
            self._poll_task = asyncio.create_task(self._poll_loop())
            if self.event_log_path:
                self._evtlog_task = asyncio.create_task(self._tail_event_log())

        async def _poll_loop(self) -> None:
            """1Hz scrape of every worker via aiperf collector."""
            assert HAVE_AIPERF, "aiperf not importable; cannot scrape"

            async def record_cb(records, collector_id: str):
                # Find the worker by collector_id
                for w in self.workers:
                    if str(w.local_port) in collector_id:
                        for rec in records:
                            update_worker_from_record(w, rec)
                        break

            async def error_cb(err: "ErrorDetails", collector_id: str):
                evt = self.query_one("#event-log", Log)
                evt.write_line(f"scrape error [{collector_id}]: {err}")

            # Init one collector per worker
            for w in self.workers:
                if not w.local_port:
                    continue
                w.collector = ServerMetricsDataCollector(
                    endpoint_url=f"http://127.0.0.1:{w.local_port}/metrics",
                    collection_interval=1.0,
                    reachability_timeout=5.0,
                    record_callback=record_cb,
                    error_callback=error_cb,
                    collector_id=f"worker_{w.local_port}",
                )
                try:
                    await w.collector.initialize()
                    await w.collector.start()
                except Exception as e:
                    self.query_one("#event-log", Log).write_line(
                        f"collector start failed [{w.pod_name}]: {e}"
                    )

            # Periodic refresh of the table from worker state
            tbl = self.query_one("#workers-table", DataTable)
            while not self.is_paused if hasattr(self, "is_paused") else True:
                for w in self.workers:
                    self._refresh_row(tbl, w)
                await asyncio.sleep(1.0)

        def _refresh_row(self, tbl, w: Worker) -> None:
            spark_w = 16
            running_cell = (
                f"{w.running:>5.0f}  {render_spark(w.history_running, spark_w)}"
            )
            waiting_cell = (
                f"{w.waiting:>5.0f}  {render_spark(w.history_waiting, spark_w)}"
            )
            kv_cell = f"{w.kv_pct:>5.1%}  {render_spark(w.history_kv, spark_w)}"
            nixl_ms = w.nixl_p99 * 1000.0 if w.nixl_p99 else 0.0
            nixl_cell = f"{nixl_ms:>5.1f}  {render_spark(w.history_nixl, spark_w)}"
            preempt_cell = f"{int(w.preempt_total)}"
            try:
                tbl.update_cell(w.pod_name, "running", running_cell)
                tbl.update_cell(w.pod_name, "waiting", waiting_cell)
                tbl.update_cell(w.pod_name, "kv%", kv_cell)
                tbl.update_cell(w.pod_name, "nixl_p99 (ms)", nixl_cell)
                tbl.update_cell(w.pod_name, "preempts", preempt_cell)
            except Exception:
                pass  # row may be transiently unavailable

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
            for w in self.workers:
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
