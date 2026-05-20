#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Local Resource Monitor — per-process resource metrics and local dashboard.

Tracks VRAM, GPU utilization, PCIe bandwidth, CPU, disk I/O, and network I/O
for Dynamo inference processes, labeled by model name and process identity.

Why this exists (full design context: DEP #9065):
  - DCGM is per-device + 1 s scrape — can't attribute VRAM/PCIe to a PID,
    and 1 s smears 10-200 ms startup events (cudaMalloc defrag, PCIe weight
    bursts, torch-compile stalls).
  - Dynamo's request-lifecycle Prometheus metrics scrape every 1-15 s and
    only fire after a request lands, so they're blind to startup failures.
  - Generic per-PID exporters don't walk the Dynamo subprocess lineage
    (pytest -> python -> bash -> python), so one test shows up as four
    unrelated rows instead of one logical worker.
  - Nsight Systems is profile-then-analyze, not always-on telemetry.
  - aiperf is client-side load+latency, not server-side per-process state,
    and runs after the engine is already up.

This exporter fills that gap: 200 ms scrape (PCIe internally 10/s),
subprocess-lineage grouping, per-engine PID labels with framework + model +
test name so Grafana can render bin-packed parallel engines as separate
side-by-side series.

Architecture (multiprocess, bypasses GIL):
  - 1 subprocess per GPU: PCIe TX/RX sampling at 10/s
  - 1 subprocess: CPU + GPU mem/util/temp + network at 5/s
  - 1 subprocess: aggregate disk I/O at 1/s
  - Main process: polls pipes, updates Prometheus gauges or pushes UI deltas

The HTTP endpoint serves /metrics for Prometheus. When the dashboard packages
are installed, the same endpoint also serves a local Plotly dashboard at / and
pushes deltas over Socket.IO.

Usage:
    python3 dynamo_local_resource_monitor.py [--port 8051] [--host 0.0.0.0]
"""

import argparse
import importlib.util
import json
import multiprocessing
import os
import signal
import socketserver
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

_REQUIRED_PACKAGES = {
    "psutil": "psutil",
    "pynvml": "nvidia-ml-py",
    "prometheus_client": "prometheus-client",
}
_DASHBOARD_PACKAGES = {
    "flask": "flask",
    "flask_socketio": "flask-socketio",
    "simple_websocket": "simple-websocket",
}
_missing = [
    pypi
    for mod, pypi in _REQUIRED_PACKAGES.items()
    if importlib.util.find_spec(mod) is None
]
if _missing:
    print(
        f"dynamo_local_resource_monitor: missing packages: {', '.join(_missing)}\n"
        f"This tool is meant to run on the HOST machine (not inside a container).\n"
        f"Install with:\n\n  pip install {' '.join(_missing)}\n",
        file=sys.stderr,
    )
    sys.exit(1)

import psutil  # noqa: E402
import pynvml  # noqa: E402
from prometheus_client import CONTENT_TYPE_LATEST, Gauge, generate_latest  # noqa: E402

HAS_NVML = False
try:
    pynvml.nvmlInit()
    HAS_NVML = True
except pynvml.NVMLError as e:
    print(f"NVML init failed ({e}) -- GPU monitoring disabled")

PROCESS_COLORS = [
    "#00e5ff",
    "#ff1744",
    "#76ff03",
    "#448aff",
    "#ff9100",
    "#d500f9",
    "#ffea00",
    "#f50057",
    "#00e676",
    "#ff6d00",
    "#18ffff",
    "#ea80fc",
    "#b2ff59",
    "#ff80ab",
    "#64ffda",
    "#ffd740",
    "#e040fb",
    "#ff9e80",
    "#a7ffeb",
    "#ff8a80",
]
OTHER_COLOR = "#9e9e9e"
CACHE_DIR = Path.home() / ".cache" / "dynamo_local_resource_monitor"
CACHE_FILE = CACHE_DIR / "metrics.json"


def parse_args():
    p = argparse.ArgumentParser(description="GPU/CPU/Disk/Network resource monitor")
    p.add_argument("--port", type=int, default=8051)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument(
        "--main-interval",
        type=int,
        default=200,
        help="Main collection interval ms (CPU + GPU mem/util/temp + network, 5/s)",
    )
    p.add_argument(
        "--disk-interval",
        type=int,
        default=1000,
        help="Disk I/O collection interval ms (1/s)",
    )
    p.add_argument(
        "--window",
        type=int,
        default=900,
        help="Dashboard rolling window seconds (default 900 = 15min)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Dashboard process series limit per chart",
    )
    return p.parse_args()


def _missing_dashboard_packages():
    return [
        pypi
        for mod, pypi in _DASHBOARD_PACKAGES.items()
        if importlib.util.find_spec(mod) is None
    ]


def _load_dashboard_server_deps():
    missing = _missing_dashboard_packages()
    if missing:
        print(
            f"dynamo_local_resource_monitor: missing dashboard server packages: {', '.join(missing)}\n"
            f"Install with:\n\n  pip install {' '.join(missing)}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    flask = importlib.import_module("flask")
    flask_socketio = importlib.import_module("flask_socketio")
    return flask, flask_socketio


def _resolve_process_name(pid: int) -> str:
    try:
        proc = psutil.Process(pid)
        cmdline = proc.cmdline()
        full_cmd = " ".join(cmdline) if cmdline else ""
        proc_name = proc.name()

        # VLLM EngineCore: walk parent chain for model name + launch script
        if proc_name == "VLLM::EngineCore" or "EngineCore" in full_cmd:
            model, script, test = _gpu_process_context(proc)
            parts = ["VLLM::EngineCore"]
            if model:
                parts.append(model)
            if test:
                parts.append(test)
            elif script:
                parts.append(script)
            return ", ".join(parts)

        # SGLang, TRT-LLM, mpi4py, orted — walk ancestry for context
        is_infra = (
            "sglang" in proc_name.lower()
            or "sglang" in full_cmd.lower()
            or "trtllm" in proc_name.lower()
            or "trtllm" in full_cmd.lower()
            or "mpi4py" in full_cmd.lower()
            or "orted" in proc_name.lower()
        )
        if is_infra:
            model, script, test = _gpu_process_context(proc)
            label = proc_name
            if "mpi4py" in full_cmd.lower():
                label = "TRT-LLM/MPI"
            elif "orted" in proc_name.lower():
                label = "TRT-LLM/orted"
            parts = [label]
            if model:
                parts.append(model)
            if test:
                parts.append(test)
            elif script:
                parts.append(script)
            return ", ".join(parts)

        # Python with dynamo module: extract module + model + test context
        if cmdline and os.path.basename(cmdline[0]).startswith("python"):
            if len(cmdline) > 1 and "-m" in cmdline:
                m_idx = cmdline.index("-m")
                if m_idx + 1 < len(cmdline):
                    module = cmdline[m_idx + 1]
                    model = _extract_arg(cmdline, "--model")
                    if not model:
                        model = _extract_arg(cmdline, "--served-model-name")
                    short_model = os.path.basename(model) if model else ""
                    _, _, test = _gpu_process_context(proc)
                    parts = [module]
                    if short_model:
                        parts.append(short_model)
                    if test:
                        parts.append(test)
                    return ", ".join(parts)
            if len(cmdline) > 1:
                script = os.path.basename(cmdline[1])
                if len(script) > 25:
                    script = script[:22] + "..."
                return script

        if cmdline:
            base = os.path.basename(cmdline[0])
            if len(base) > 25:
                base = base[:22] + "..."
            return base
        return proc_name
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return f"pid:{pid}"


def _gpu_process_context(proc) -> tuple[str, str, str]:
    """Walk parent chain to find model name, launch script, and test name."""
    model = ""
    script = ""
    test = ""
    try:
        for ancestor in proc.parents():
            acmd = ancestor.cmdline()
            if not acmd:
                continue
            if not model:
                for flag in ("--model", "--model-path", "--served-model-name"):
                    m = _extract_arg(acmd, flag)
                    if m:
                        model = os.path.basename(m)
                        break
            if not script:
                for arg in acmd:
                    if arg.endswith(".sh"):
                        script = os.path.basename(arg)
                        break
            if not test:
                full = " ".join(acmd)
                if "pytest" in full or "py.test" in full:
                    for arg in acmd:
                        if "::" in arg and ".py" in arg:
                            test = arg.split("::")[-1]
                            break
            if model and script and test:
                break
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return model, script, test


def _extract_arg(cmdline: list[str], flag: str) -> str:
    """Extract the value after a CLI flag like --model. Also checks --model-path.

    vLLM uses ``--model``, SGLang uses ``--model-path`` for the same thing —
    the function tries both so the same caller works regardless of backend.
    """
    for f in (flag, flag + "-path"):
        try:
            idx = cmdline.index(f)
            if idx + 1 < len(cmdline):
                return cmdline[idx + 1]
        except ValueError:
            pass
    return ""


class ProcessTracker:
    """Track rolling per-process series with stable colors for the dashboard UI."""

    def __init__(self, maxlen: int, prune: bool = True):
        self.maxlen = maxlen
        self._prune_enabled = prune
        self._len = 0
        self.series: dict[int, deque[float]] = {}
        self.names: dict[int, str] = {}
        self.first_seen: dict[int, float] = {}
        self._pid_slot: dict[int, int] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0

    def new_pids(self, data: dict[int, float]) -> set[int]:
        return set(data.keys()) - set(self.series.keys())

    def record(
        self,
        data: dict[int, float],
        name_resolver,
        timestamp: float = 0,
        pre_resolved: dict[int, str] | None = None,
    ):
        for pid in data:
            if pid not in self.series:
                backfill = min(self._len, self.maxlen)
                self.series[pid] = deque([0.0] * backfill, maxlen=self.maxlen)
                if pre_resolved and pid in pre_resolved:
                    self.names[pid] = pre_resolved[pid]
                else:
                    self.names[pid] = name_resolver(pid)
                self.first_seen[pid] = timestamp
                if self._free_slots:
                    slot = self._free_slots.pop(0)
                else:
                    slot = self._next_slot
                    self._next_slot += 1
                self._pid_slot[pid] = slot

        for pid, dq in self.series.items():
            dq.append(data.get(pid, 0.0))
        self._len += 1
        if self._prune_enabled:
            self._prune_dead()

    def _prune_dead(self):
        window = min(200, self._len)
        dead = [
            pid
            for pid, dq in self.series.items()
            if all(v == 0.0 for v in list(dq)[-window:])
        ]
        for pid in dead:
            del self.series[pid]
            del self.names[pid]
            if pid in self._pid_slot:
                self._free_slots.append(self._pid_slot.pop(pid))

    def to_dict(self) -> dict:
        return {
            "maxlen": self.maxlen,
            "_prune_enabled": self._prune_enabled,
            "_len": self._len,
            "series": {str(k): list(v) for k, v in self.series.items()},
            "names": {str(k): v for k, v in self.names.items()},
            "first_seen": {str(k): v for k, v in self.first_seen.items()},
            "_pid_slot": {str(k): v for k, v in self._pid_slot.items()},
            "_free_slots": self._free_slots,
            "_next_slot": self._next_slot,
        }

    def load_dict(self, data: dict):
        self._len = data.get("_len", 0)
        for key, values in data.get("series", {}).items():
            pid = int(key)
            self.series[pid] = deque(values, maxlen=self.maxlen)
        self.names = {int(k): v for k, v in data.get("names", {}).items()}
        self.first_seen = {int(k): v for k, v in data.get("first_seen", {}).items()}
        self._pid_slot = {int(k): v for k, v in data.get("_pid_slot", {}).items()}
        self._free_slots = data.get("_free_slots", [])
        self._next_slot = data.get("_next_slot", 0)

    def _color_for(self, pid: int) -> str:
        slot = self._pid_slot.get(pid, 0)
        return PROCESS_COLORS[slot % len(PROCESS_COLORS)]

    def get_top_sorted(
        self, n: int = 20, sort_by: str = "recency"
    ) -> list[tuple[int, str, str, list[float]]]:
        if not self.series:
            return []

        if sort_by == "recency":

            def _key(p):
                series = self.series[p]
                last_active = -1
                for i in range(len(series) - 1, -1, -1):
                    if series[i] > 0:
                        last_active = i
                        break
                return (last_active, max(series) if series else 0)

        else:

            def _key(p):
                return sum(self.series[p])

        sorted_pids = sorted(self.series.keys(), key=_key, reverse=True)
        top = sorted_pids[:n]
        rest = sorted_pids[n:]
        result = [
            (pid, self.names[pid], self._color_for(pid), list(self.series[pid]))
            for pid in top
            if max(self.series[pid]) > 0
        ]
        if rest:
            length = len(next(iter(self.series.values())))
            rest_vals = [0.0] * length
            has_data = False
            for pid in rest:
                for i, value in enumerate(self.series[pid]):
                    rest_vals[i] += value
                    if value > 0:
                        has_data = True
            if has_data:
                result.append((-1, "Other", OTHER_COLOR, rest_vals))
        return result


# ---------------------------------------------------------------------------
# Multiprocess workers (each gets its own GIL)
# ---------------------------------------------------------------------------


def _pcie_worker(conn, gpu_idx):
    """Subprocess: sample PCIe TX/RX for one GPU at ~10/s."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    TX = pynvml.NVML_PCIE_UTIL_TX_BYTES
    RX = pynvml.NVML_PCIE_UTIL_RX_BYTES
    while True:
        now = time.time()
        try:
            tx = pynvml.nvmlDeviceGetPcieThroughput(handle, TX) / (1024.0 * 1024.0)
            rx = pynvml.nvmlDeviceGetPcieThroughput(handle, RX) / (1024.0 * 1024.0)
        except pynvml.NVMLError:
            tx, rx = 0.0, 0.0
        conn.send(("pcie", now, gpu_idx, tx, rx))
        time.sleep(0.1)


def _main_worker(conn, gpu_count, interval_sec):
    """Subprocess: CPU (overall + per-name groups) + GPU mem/util + network at 5/s; temperature at 1/s."""
    handles = []
    if gpu_count > 0:
        try:
            pynvml.nvmlInit()
            handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]
        except pynvml.NVMLError as exc:
            print(
                f"[main_worker] NVML init failed ({exc}), GPU metrics disabled",
                flush=True,
            )
            gpu_count = 0
    psutil.cpu_percent(interval=None)
    # Prime per-process cpu_percent so subsequent calls return real deltas
    list(psutil.process_iter(["name", "cpu_percent"]))
    ncpus = psutil.cpu_count() or 1
    prev_net = psutil.net_io_counters()
    last_mono = time.monotonic()
    # Temperature changes slowly; sample at ~1/s by collecting every temp_every-th cycle.
    temp_every = max(1, round(1 / interval_sec))
    cycle = 0
    prev_temps: list[float] = [0.0] * gpu_count
    while True:
        now = time.time()
        mono = time.monotonic()
        dt = mono - last_mono
        if dt <= 0:
            dt = interval_sec
        cpu = psutil.cpu_percent(interval=None)
        # Per-process CPU grouped by name, normalised to total CPU capacity (0-100%)
        cpu_by_name: dict[str, float] = {}
        for p in psutil.process_iter(["name", "cpu_percent"]):
            try:
                name = p.info["name"] or "?"
                pct = p.info["cpu_percent"] or 0.0
                if pct > 0:
                    cpu_by_name[name] = cpu_by_name.get(name, 0.0) + pct
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        for k in cpu_by_name:
            cpu_by_name[k] /= ncpus
        collect_temp = cycle % temp_every == 0
        gpu_data = []
        for gi, h in enumerate(handles):
            proc_mem: dict[int, float] = {}
            for getter in (
                pynvml.nvmlDeviceGetComputeRunningProcesses,
                pynvml.nvmlDeviceGetGraphicsRunningProcesses,
            ):
                try:
                    for p in getter(h):
                        mem_gib = (p.usedGpuMemory or 0) / (1024**3)
                        proc_mem[p.pid] = proc_mem.get(p.pid, 0.0) + mem_gib
                except pynvml.NVMLError:
                    pass
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except pynvml.NVMLError:
                util = 0.0
            if collect_temp:
                try:
                    prev_temps[gi] = float(
                        pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    )
                except pynvml.NVMLError:
                    pass
            gpu_data.append((proc_mem, util, prev_temps[gi]))
        net = psutil.net_io_counters()
        sent_rate = (net.bytes_sent - prev_net.bytes_sent) / (1024 * 1024) / dt
        recv_rate = (net.bytes_recv - prev_net.bytes_recv) / (1024 * 1024) / dt
        prev_net = net
        last_mono = mono
        conn.send(("main", now, cpu, gpu_data, sent_rate, recv_rate, cpu_by_name))
        cycle += 1
        time.sleep(interval_sec)


def _disk_worker(conn, interval_sec):
    """Subprocess: aggregate disk I/O (1/s)."""
    prev = psutil.disk_io_counters()
    last_mono = time.monotonic()
    while True:
        now = time.time()
        mono = time.monotonic()
        dt = mono - last_mono
        if dt <= 0:
            dt = interval_sec
        cur = psutil.disk_io_counters()
        read_mbps = (cur.read_bytes - prev.read_bytes) / (1024 * 1024) / dt
        write_mbps = (cur.write_bytes - prev.write_bytes) / (1024 * 1024) / dt
        prev = cur
        last_mono = mono
        conn.send(("disk", now, read_mbps, write_mbps))
        time.sleep(interval_sec)


class MetricsCollector:
    """Rolling time-series state for the WebSocket dashboard."""

    def __init__(
        self,
        window_sec: int,
        main_interval_ms: int = 200,
        disk_interval_ms: int = 1000,
        pcie_rate_hz: int = 10,
        top_n: int = 12,
    ):
        self.lock = threading.Lock()
        maxlen_main = int(window_sec * 1000 / main_interval_ms)
        maxlen_pcie = window_sec * pcie_rate_hz
        maxlen_disk = int(window_sec * 1000 / disk_interval_ms)
        self.top_n = top_n

        self.ts_main: deque[float] = deque(maxlen=maxlen_main)
        self.counter_main = 0
        self.cpu_pct: deque[float] = deque(maxlen=maxlen_main)
        self.proc_gpu_mem: list[ProcessTracker] = []
        self.gpu_util: list[deque[float]] = []
        self.gpu_temp: list[deque[float]] = []
        self.net_sent_mbps: deque[float] = deque(maxlen=maxlen_main)
        self.net_recv_mbps: deque[float] = deque(maxlen=maxlen_main)

        self.ts_pcie: list[deque[float]] = []
        self.counter_pcie: list[int] = []
        self.gpu_pcie_tx: list[deque[float]] = []
        self.gpu_pcie_rx: list[deque[float]] = []
        self.has_pcie = False

        self.proc_cpu = ProcessTracker(maxlen_main, prune=True)
        self._cpu_name_to_id: dict[str, int] = {}
        self._cpu_next_id = 1
        self._cpu_top_n = 5

        self.ts_disk: deque[float] = deque(maxlen=maxlen_disk)
        self.counter_disk = 0
        self.disk_read_mbps: deque[float] = deque(maxlen=maxlen_disk)
        self.disk_write_mbps: deque[float] = deque(maxlen=maxlen_disk)

        self.gpu_count = 0
        self.gpu_names: list[str] = []
        self.gpu_mem_total_gib: list[float] = []
        if HAS_NVML:
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")
                self.gpu_names.append(name)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.gpu_mem_total_gib.append(mem.total / (1024**3))
                self.proc_gpu_mem.append(ProcessTracker(maxlen_main, prune=False))
                self.gpu_util.append(deque(maxlen=maxlen_main))
                self.gpu_temp.append(deque(maxlen=maxlen_main))
                self.gpu_pcie_tx.append(deque(maxlen=maxlen_pcie))
                self.gpu_pcie_rx.append(deque(maxlen=maxlen_pcie))
                self.ts_pcie.append(deque(maxlen=maxlen_pcie))
                self.counter_pcie.append(0)
            if self.gpu_count > 0:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                    )
                    self.has_pcie = True
                except pynvml.NVMLError:
                    self.has_pcie = False

    def save_state(self):
        """Persist the rolling window so restarting the dashboard keeps context."""
        with self.lock:
            state = {
                "ts_main": list(self.ts_main),
                "counter_main": self.counter_main,
                "cpu_pct": list(self.cpu_pct),
                "proc_cpu": self.proc_cpu.to_dict(),
                "_cpu_name_to_id": self._cpu_name_to_id,
                "_cpu_next_id": self._cpu_next_id,
                "net_sent_mbps": list(self.net_sent_mbps),
                "net_recv_mbps": list(self.net_recv_mbps),
                "ts_pcie": [list(d) for d in self.ts_pcie],
                "counter_pcie": self.counter_pcie[:],
                "gpu_pcie_tx": [list(d) for d in self.gpu_pcie_tx],
                "gpu_pcie_rx": [list(d) for d in self.gpu_pcie_rx],
                "ts_disk": list(self.ts_disk),
                "counter_disk": self.counter_disk,
                "disk_read_mbps": list(self.disk_read_mbps),
                "disk_write_mbps": list(self.disk_write_mbps),
                "gpu_util": [list(d) for d in self.gpu_util],
                "gpu_temp": [list(d) for d in self.gpu_temp],
                "proc_gpu_mem": [t.to_dict() for t in self.proc_gpu_mem],
                "gpu_count": self.gpu_count,
                "saved_at": time.time(),
            }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.rename(CACHE_FILE)
        print(f"[save] metrics saved to {CACHE_FILE}", flush=True)

    def load_state(self):
        """Restore the previous rolling window, if it matches the local GPUs."""
        if not CACHE_FILE.exists():
            return
        try:
            state = json.loads(CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[load] failed to read cache: {exc}", flush=True)
            return
        cached_gpu_count = state.get("gpu_count")
        if cached_gpu_count != self.gpu_count:
            print(
                f"[load] GPU count mismatch (cached={cached_gpu_count}, "
                f"current={self.gpu_count}), ignoring cache",
                flush=True,
            )
            return
        age = time.time() - state.get("saved_at", 0)
        print(
            f"[load] restoring metrics from {CACHE_FILE} (saved {age:.0f}s ago)",
            flush=True,
        )

        def state_list(name: str, index: int, default):
            values = state.get(name, [])
            if isinstance(values, list) and index < len(values):
                return values[index]
            return default

        with self.lock:
            maxlen_main = self.ts_main.maxlen
            maxlen_pcie = self.ts_pcie[0].maxlen if self.ts_pcie else 0
            maxlen_disk = self.ts_disk.maxlen
            self.ts_main = deque(state.get("ts_main", []), maxlen=maxlen_main)
            self.counter_main = state.get("counter_main", 0)
            self.cpu_pct = deque(state.get("cpu_pct", []), maxlen=maxlen_main)
            if "proc_cpu" in state:
                self.proc_cpu.load_dict(state["proc_cpu"])
                self._cpu_name_to_id = state.get("_cpu_name_to_id", {})
                self._cpu_next_id = state.get("_cpu_next_id", 1)
            self.net_sent_mbps = deque(
                state.get("net_sent_mbps", []), maxlen=maxlen_main
            )
            self.net_recv_mbps = deque(
                state.get("net_recv_mbps", []), maxlen=maxlen_main
            )
            for gi in range(self.gpu_count):
                self.ts_pcie[gi] = deque(
                    state_list("ts_pcie", gi, []), maxlen=maxlen_pcie
                )
                self.counter_pcie[gi] = state_list("counter_pcie", gi, 0)
                self.gpu_pcie_tx[gi] = deque(
                    state_list("gpu_pcie_tx", gi, []), maxlen=maxlen_pcie
                )
                self.gpu_pcie_rx[gi] = deque(
                    state_list("gpu_pcie_rx", gi, []), maxlen=maxlen_pcie
                )
                self.gpu_util[gi] = deque(
                    state_list("gpu_util", gi, []), maxlen=maxlen_main
                )
                self.gpu_temp[gi] = deque(
                    state_list("gpu_temp", gi, []), maxlen=maxlen_main
                )
                self.proc_gpu_mem[gi].load_dict(state_list("proc_gpu_mem", gi, {}))
            self.ts_disk = deque(state.get("ts_disk", []), maxlen=maxlen_disk)
            self.counter_disk = state.get("counter_disk", 0)
            self.disk_read_mbps = deque(
                state.get("disk_read_mbps", []), maxlen=maxlen_disk
            )
            self.disk_write_mbps = deque(
                state.get("disk_write_mbps", []), maxlen=maxlen_disk
            )
        print(
            f"[load] restored {len(self.ts_main)} main samples, {len(self.ts_disk)} disk samples",
            flush=True,
        )

    def ingest_pcie(self, now: float, gpu_idx: int, tx_val: float, rx_val: float):
        with self.lock:
            self.ts_pcie[gpu_idx].append(now)
            self.counter_pcie[gpu_idx] += 1
            self.gpu_pcie_tx[gpu_idx].append(tx_val)
            self.gpu_pcie_rx[gpu_idx].append(rx_val)

    def _cpu_name_id(self, name: str) -> int:
        if name not in self._cpu_name_to_id:
            self._cpu_name_to_id[name] = self._cpu_next_id
            self._cpu_next_id += 1
        return self._cpu_name_to_id[name]

    def ingest_main(
        self,
        now: float,
        cpu: float,
        gpu_data: list[tuple[dict, float, float]],
        sent_rate: float,
        recv_rate: float,
        cpu_by_name: dict[str, float] | None = None,
    ):
        gpu_names: list[dict[int, str]] = []
        for i, (gpu_proc_mem, _, _) in enumerate(gpu_data):
            new_pids = self.proc_gpu_mem[i].new_pids(gpu_proc_mem)
            gpu_names.append({pid: _resolve_process_name(pid) for pid in new_pids})

        cpu_id_data: dict[int, float] = {}
        cpu_id_names: dict[int, str] = {}
        if cpu_by_name:
            for name, pct in cpu_by_name.items():
                sid = self._cpu_name_id(name)
                cpu_id_data[sid] = pct
                cpu_id_names[sid] = name

        with self.lock:
            self.ts_main.append(now)
            self.counter_main += 1
            self.cpu_pct.append(cpu)
            self.proc_cpu.record(
                cpu_id_data,
                lambda sid: cpu_id_names.get(sid, f"pid:{sid}"),
                now,
                pre_resolved=cpu_id_names,
            )
            for i, (gpu_proc_mem, util, temp) in enumerate(gpu_data):
                self.proc_gpu_mem[i].record(
                    gpu_proc_mem,
                    _resolve_process_name,
                    now,
                    pre_resolved=gpu_names[i],
                )
                self.gpu_util[i].append(util)
                self.gpu_temp[i].append(temp)
            alpha_net = 0.15
            prev_sent = self.net_sent_mbps[-1] if self.net_sent_mbps else 0.0
            prev_recv = self.net_recv_mbps[-1] if self.net_recv_mbps else 0.0
            self.net_sent_mbps.append(
                alpha_net * sent_rate + (1 - alpha_net) * prev_sent
            )
            self.net_recv_mbps.append(
                alpha_net * recv_rate + (1 - alpha_net) * prev_recv
            )

    def ingest_disk(self, now: float, read_mbps: float, write_mbps: float):
        with self.lock:
            self.ts_disk.append(now)
            self.counter_disk += 1
            self.disk_read_mbps.append(read_mbps)
            self.disk_write_mbps.append(write_mbps)

    @staticmethod
    def _downsample_init(times, arrays, max_recent=1000, max_total=1500):
        n = len(times)
        if n <= max_total:
            return times, arrays
        older_count = n - max_recent
        older_target = max_total - max_recent
        step = max(1, older_count // older_target)
        indices = list(range(0, older_count, step)) + list(range(older_count, n))
        ds_times = [times[i] for i in indices]
        ds_arrays = [
            [arr[i] for i in indices] if len(arr) == n else arr for arr in arrays
        ]
        return ds_times, ds_arrays

    def snapshot_full(self) -> dict:
        with self.lock:
            main_vals = [list(self.cpu_pct)]
            cpu_procs_full = self.proc_cpu.get_top_sorted(
                self._cpu_top_n, sort_by="total"
            )
            for _, _, _, values in cpu_procs_full:
                main_vals.append(values)

            gpu_mem_per_gpu_full = []
            for gi in range(self.gpu_count):
                procs = self.proc_gpu_mem[gi].get_top_sorted(self.top_n)
                gpu_mem_per_gpu_full.append(procs)
                for _, _, _, values in procs:
                    main_vals.append(values)

            gpu_util = [list(self.gpu_util[i]) for i in range(self.gpu_count)]
            gpu_temp = [list(self.gpu_temp[i]) for i in range(self.gpu_count)]
            main_vals.extend(gpu_util)
            main_vals.extend(gpu_temp)
            main_vals.append(list(self.net_sent_mbps))
            main_vals.append(list(self.net_recv_mbps))
            ts_main_ds, main_ds = self._downsample_init(list(self.ts_main), main_vals)

            midx = 0
            ds_cpu = main_ds[midx] if main_ds else []
            midx += 1
            cpu_procs = []
            for proc_id, name, color, _ in cpu_procs_full:
                cpu_procs.append(
                    {
                        "id": proc_id,
                        "name": name,
                        "color": color,
                        "vals": main_ds[midx],
                    }
                )
                midx += 1

            gpu_mem_per_gpu = []
            for gi in range(self.gpu_count):
                ds_procs = []
                for pid, name, color, _ in gpu_mem_per_gpu_full[gi]:
                    ds_procs.append(
                        {
                            "pid": pid,
                            "name": name,
                            "color": color,
                            "vals": main_ds[midx],
                            "first_seen": self.proc_gpu_mem[gi].first_seen.get(pid, 0),
                        }
                    )
                    midx += 1
                gpu_mem_per_gpu.append(ds_procs)

            ds_gpu_util = [main_ds[midx + i] for i in range(self.gpu_count)]
            midx += self.gpu_count
            ds_gpu_temp = [main_ds[midx + i] for i in range(self.gpu_count)]
            midx += self.gpu_count
            ds_net_sent = main_ds[midx] if midx < len(main_ds) else []
            ds_net_recv = main_ds[midx + 1] if midx + 1 < len(main_ds) else []

            ds_pcie_tx = []
            ds_pcie_rx = []
            ts_pcie_per_gpu = []
            if self.has_pcie:
                for gi in range(self.gpu_count):
                    ts_raw = list(self.ts_pcie[gi])
                    tx_raw = list(self.gpu_pcie_tx[gi])
                    rx_raw = list(self.gpu_pcie_rx[gi])
                    ts_ds, vals_ds = self._downsample_init(ts_raw, [tx_raw, rx_raw])
                    ts_pcie_per_gpu.append(ts_ds)
                    ds_pcie_tx.append(vals_ds[0])
                    ds_pcie_rx.append(vals_ds[1])

            disk_vals = [list(self.disk_read_mbps), list(self.disk_write_mbps)]
            ts_disk_ds, disk_ds = self._downsample_init(list(self.ts_disk), disk_vals)
            ds_disk_read = disk_ds[0] if disk_ds else []
            ds_disk_write = disk_ds[1] if len(disk_ds) > 1 else []

            gpu_summary = ""
            if self.gpu_names:
                parts = []
                for name, count in Counter(self.gpu_names).items():
                    short = name.replace("NVIDIA ", "").replace("Generation", "")
                    short = short.strip()
                    parts.append(f"{count}x {short}" if count > 1 else short)
                gpu_summary = "[" + ", ".join(parts) + "]"

            return {
                "type": "init",
                "counter_main": self.counter_main,
                "counter_pcie": list(self.counter_pcie),
                "counter_disk": self.counter_disk,
                "ts_main": ts_main_ds,
                "ts_pcie": ts_pcie_per_gpu,
                "ts_disk": ts_disk_ds,
                "hostname": os.uname().nodename,
                "gpu_summary": gpu_summary,
                "gpu_count": self.gpu_count,
                "gpu_names": self.gpu_names,
                "gpu_mem_total_gib": self.gpu_mem_total_gib,
                "has_pcie": self.has_pcie,
                "gpu_mem": gpu_mem_per_gpu,
                "gpu_util": ds_gpu_util,
                "gpu_temp": ds_gpu_temp,
                "pcie_tx": ds_pcie_tx,
                "pcie_rx": ds_pcie_rx,
                "cpu": ds_cpu,
                "cpu_procs": cpu_procs,
                "net_sent": ds_net_sent,
                "net_recv": ds_net_recv,
                "disk_read": ds_disk_read,
                "disk_write": ds_disk_write,
            }

    def snapshot_delta(
        self,
        since_main: int,
        since_pcie: list[int],
        since_disk: int,
        step: int = 1,
    ) -> dict:
        with self.lock:
            new_main = self.counter_main - since_main
            main_len = len(self.ts_main)
            main_idx = max(0, main_len - new_main) if new_main > 0 else main_len
            sl_m = slice(main_idx, None, step)
            new_ts_main = list(self.ts_main)[sl_m]
            cpu = list(self.cpu_pct)[sl_m]

            cpu_procs = self.proc_cpu.get_top_sorted(self._cpu_top_n, sort_by="total")
            cpu_procs_delta = [
                {
                    "id": proc_id,
                    "name": name,
                    "color": color,
                    "vals": values[sl_m] if main_idx < len(values) else [],
                }
                for proc_id, name, color, values in cpu_procs
            ]
            cpu_proc_keys = [proc_id for proc_id, _, _, _ in cpu_procs]

            gpu_mem_per_gpu = []
            gpu_mem_keys_per_gpu = []
            for gi in range(self.gpu_count):
                gpu_mem = self.proc_gpu_mem[gi].get_top_sorted(self.top_n)
                gpu_mem_per_gpu.append(
                    [
                        {
                            "pid": pid,
                            "name": name,
                            "color": color,
                            "vals": values[sl_m] if main_idx < len(values) else [],
                        }
                        for pid, name, color, values in gpu_mem
                    ]
                )
                gpu_mem_keys_per_gpu.append([pid for pid, _, _, _ in gpu_mem])

            gpu_util = [list(self.gpu_util[i])[sl_m] for i in range(self.gpu_count)]
            gpu_temp = [list(self.gpu_temp[i])[sl_m] for i in range(self.gpu_count)]
            net_sent = list(self.net_sent_mbps)[sl_m]
            net_recv = list(self.net_recv_mbps)[sl_m]

            pcie_step = max(step, 2)
            pcie_tx = []
            pcie_rx = []
            new_ts_pcie = []
            if self.has_pcie:
                for gi in range(self.gpu_count):
                    sp = since_pcie[gi] if gi < len(since_pcie) else 0
                    new_pcie = self.counter_pcie[gi] - sp
                    pcie_len = len(self.ts_pcie[gi])
                    pcie_idx = max(0, pcie_len - new_pcie) if new_pcie > 0 else pcie_len
                    sl_p = slice(pcie_idx, None, pcie_step)
                    new_ts_pcie.append(list(self.ts_pcie[gi])[sl_p])
                    pcie_tx.append(list(self.gpu_pcie_tx[gi])[sl_p])
                    pcie_rx.append(list(self.gpu_pcie_rx[gi])[sl_p])

            new_disk = self.counter_disk - since_disk
            disk_len = len(self.ts_disk)
            disk_idx = max(0, disk_len - new_disk) if new_disk > 0 else disk_len
            sl_d = slice(disk_idx, None, step)
            new_ts_disk = list(self.ts_disk)[sl_d]
            disk_read = list(self.disk_read_mbps)[sl_d]
            disk_write = list(self.disk_write_mbps)[sl_d]

            return {
                "type": "delta",
                "counter_main": self.counter_main,
                "counter_pcie": list(self.counter_pcie),
                "counter_disk": self.counter_disk,
                "ts_main": new_ts_main,
                "ts_pcie": new_ts_pcie,
                "ts_disk": new_ts_disk,
                "gpu_mem": gpu_mem_per_gpu,
                "gpu_mem_keys": gpu_mem_keys_per_gpu,
                "gpu_util": gpu_util,
                "gpu_temp": gpu_temp,
                "pcie_tx": pcie_tx,
                "pcie_rx": pcie_rx,
                "cpu": cpu,
                "cpu_procs": cpu_procs_delta,
                "cpu_proc_keys": cpu_proc_keys,
                "net_sent": net_sent,
                "net_recv": net_recv,
                "disk_read": disk_read,
                "disk_write": disk_write,
            }


# ---------------------------------------------------------------------------
# Prometheus gauges
# ---------------------------------------------------------------------------

# GPU metrics (labeled by gpu index)
GPU_MEM_USED_GIB = Gauge(
    "dynamo_gpu_memory_used_gib",
    "GPU memory used by process (GiB)",
    ["gpu", "pid", "process_name"],
)
GPU_MEM_TOTAL_GIB = Gauge(
    "dynamo_gpu_memory_total_gib",
    "Total GPU memory (GiB)",
    ["gpu"],
)
GPU_UTIL_PCT = Gauge(
    "dynamo_gpu_utilization_percent",
    "GPU utilization (%)",
    ["gpu"],
)
GPU_TEMP_C = Gauge(
    "dynamo_gpu_temperature_celsius",
    "GPU temperature (Celsius)",
    ["gpu"],
)
GPU_PCIE_TX_GBPS = Gauge(
    "dynamo_gpu_pcie_tx_gbps",
    "GPU PCIe TX throughput (GB/s)",
    ["gpu"],
)
GPU_PCIE_RX_GBPS = Gauge(
    "dynamo_gpu_pcie_rx_gbps",
    "GPU PCIe RX throughput (GB/s)",
    ["gpu"],
)

# CPU metrics
CPU_UTIL_PCT = Gauge(
    "dynamo_cpu_utilization_percent",
    "Overall CPU utilization (%)",
)
CPU_PROC_PCT = Gauge(
    "dynamo_cpu_process_percent",
    "CPU usage by process name group (%)",
    ["process_name"],
)

# Network metrics
NET_SENT_MBPS = Gauge(
    "dynamo_network_sent_mbps",
    "Network bytes sent (MB/s)",
)
NET_RECV_MBPS = Gauge(
    "dynamo_network_recv_mbps",
    "Network bytes received (MB/s)",
)

# Disk metrics
DISK_READ_MBPS = Gauge(
    "dynamo_disk_read_mbps",
    "Disk read throughput (MB/s)",
)
DISK_WRITE_MBPS = Gauge(
    "dynamo_disk_write_mbps",
    "Disk write throughput (MB/s)",
)


# ---------------------------------------------------------------------------
# Gauge updater (runs in main process, reads from worker pipes)
# ---------------------------------------------------------------------------


class PrometheusUpdater:
    """Receives data from multiprocess workers and updates Prometheus gauges."""

    def __init__(self, gpu_count: int):
        self.gpu_count = gpu_count
        # Track known GPU process PIDs so we can clear stale ones
        self._gpu_proc_pids: list[set[int]] = [set() for _ in range(gpu_count)]
        self._gpu_proc_names: dict[int, str] = {}

    def ingest_pcie(self, gpu_idx: int, tx_val: float, rx_val: float):
        gi = str(gpu_idx)
        GPU_PCIE_TX_GBPS.labels(gpu=gi).set(tx_val)
        GPU_PCIE_RX_GBPS.labels(gpu=gi).set(rx_val)

    def ingest_main(
        self,
        cpu: float,
        gpu_data: list[tuple[dict, float, float]],
        sent_rate: float,
        recv_rate: float,
        cpu_by_name: dict[str, float] | None = None,
    ):
        CPU_UTIL_PCT.set(cpu)
        NET_SENT_MBPS.set(sent_rate)
        NET_RECV_MBPS.set(recv_rate)

        # Per-process-name CPU usage. Use `is not None` so that an empty dict
        # still clears the prior cycle's labels — otherwise a tick with zero
        # active processes would leak stale per-process CPU values.
        if cpu_by_name is not None:
            CPU_PROC_PCT.clear()
            for name, pct in cpu_by_name.items():
                if pct > 0.1:
                    CPU_PROC_PCT.labels(process_name=name).set(pct)

        # Per-GPU metrics
        for gi, (proc_mem, util, temp) in enumerate(gpu_data):
            gi_str = str(gi)
            GPU_UTIL_PCT.labels(gpu=gi_str).set(util)
            GPU_TEMP_C.labels(gpu=gi_str).set(temp)

            # Resolve names for new PIDs outside the lock
            new_pids = set(proc_mem.keys()) - self._gpu_proc_pids[gi]
            for pid in new_pids:
                self._gpu_proc_names[pid] = _resolve_process_name(pid)

            # Clear stale per-process memory gauges for this GPU, then set current
            stale = self._gpu_proc_pids[gi] - set(proc_mem.keys())
            for pid in stale:
                name = self._gpu_proc_names.get(pid, f"pid:{pid}")
                GPU_MEM_USED_GIB.remove(gi_str, str(pid), name)
                self._gpu_proc_names.pop(pid, None)
            self._gpu_proc_pids[gi] = set(proc_mem.keys())

            for pid, mem_gib in proc_mem.items():
                name = self._gpu_proc_names.get(pid, f"pid:{pid}")
                GPU_MEM_USED_GIB.labels(
                    gpu=gi_str, pid=str(pid), process_name=name
                ).set(mem_gib)

    def ingest_disk(self, read_mbps: float, write_mbps: float):
        DISK_READ_MBPS.set(read_mbps)
        DISK_WRITE_MBPS.set(write_mbps)


DASHBOARD_TEMPLATE = Path(__file__).with_name("dynamo_local_resource_monitor.html.j2")


class ThreadingWSGIServer(socketserver.ThreadingMixIn, WSGIServer):
    daemon_threads = True
    allow_reuse_address = True


class QuietWSGIRequestHandler(WSGIRequestHandler):
    def log_message(self, _format, *args):
        return


def _method_not_allowed(start_response):
    body = b"Method Not Allowed\n"
    start_response(
        "405 Method Not Allowed",
        [
            ("Allow", "GET, OPTIONS"),
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]


def _dashboard_install_html(missing: list[str]) -> bytes:
    install_cmd = f"pip install {' '.join(missing)}"
    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dynamo Local Resource Monitor</title>
<style>
  body {{ margin: 32px; font-family: monospace; line-height: 1.5; }}
  code {{ background: #f3f3f3; padding: 2px 4px; }}
</style>
</head>
<body>
<h1>Dynamo Local Resource Monitor</h1>
<p>You need to install the dashboard packages to use this page:</p>
<pre>{install_cmd}</pre>
<p>Prometheus metrics are still available at <code>/metrics</code>.</p>
</body>
</html>
"""
    return body.encode("utf-8")


def start_metrics_only_server(port: int, addr: str, dashboard_missing: list[str]):
    install_html = _dashboard_install_html(dashboard_missing)

    def app(environ, start_response):
        method = environ["REQUEST_METHOD"]
        path = environ.get("PATH_INFO", "/")
        if method == "OPTIONS":
            start_response("200 OK", [("Allow", "OPTIONS,GET")])
            return [b""]
        if method != "GET":
            return _method_not_allowed(start_response)

        if path == "/metrics":
            output = generate_latest()
            start_response(
                "200 OK",
                [
                    ("Content-Type", CONTENT_TYPE_LATEST),
                    ("Content-Length", str(len(output))),
                ],
            )
            return [output]

        if path == "/favicon.ico":
            start_response("200 OK", [("Content-Length", "0")])
            return [b""]

        if path == "/":
            start_response(
                "200 OK",
                [
                    ("Content-Type", "text/html; charset=utf-8"),
                    ("Content-Length", str(len(install_html))),
                ],
            )
            return [install_html]

        body = b"Not Found\n"
        start_response(
            "404 Not Found",
            [
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", str(len(body))),
            ],
        )
        return [body]

    return make_server(
        addr,
        port,
        app,
        server_class=ThreadingWSGIServer,
        handler_class=QuietWSGIRequestHandler,
    )


def build_monitor_server(collector: MetricsCollector):
    flask, flask_socketio = _load_dashboard_server_deps()
    app = flask.Flask(__name__, template_folder=str(DASHBOARD_TEMPLATE.parent))
    app.config["SECRET_KEY"] = "dynamo_local_resource_monitor"
    socketio = flask_socketio.SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        ping_timeout=60,
        ping_interval=25,
        max_http_buffer_size=50 * 1024 * 1024,
    )

    ui_state = {"view_minutes": 2}

    @app.route("/")
    def index():
        return flask.render_template(DASHBOARD_TEMPLATE.name)

    @app.route("/metrics")
    def metrics():
        return flask.Response(
            generate_latest(),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    @socketio.on("connect")
    def handle_connect():
        sid = flask.request.sid

        def send_init():
            for _ in range(50):
                with collector.lock:
                    if len(collector.ts_main) > 5:
                        break
                time.sleep(0.1)
            socketio.emit("init", collector.snapshot_full(), to=sid)

        threading.Thread(target=send_init, daemon=True).start()

    @socketio.on("request_init")
    def handle_request_init():
        socketio.emit("init", collector.snapshot_full(), to=flask.request.sid)

    @socketio.on("view_minutes")
    def handle_view_minutes(view_minutes):
        ui_state["view_minutes"] = int(view_minutes)

    push_policy = {
        1: (0.075, 1),
        2: (0.100, 1),
        5: (0.200, 2),
        10: (0.500, 4),
        15: (1.000, 8),
    }

    def push_loop():
        last_main = 0
        last_pcie: list[int] = [0] * collector.gpu_count
        last_disk = 0
        while True:
            view_minutes = ui_state.get("view_minutes", 2)
            push_sec, step = push_policy.get(view_minutes, (0.100, 1))
            time.sleep(push_sec)
            with collector.lock:
                counter_main = collector.counter_main
                counter_pcie = list(collector.counter_pcie)
                counter_disk = collector.counter_disk
            if (
                counter_main <= last_main
                and counter_pcie == last_pcie
                and counter_disk <= last_disk
            ):
                continue
            delta = collector.snapshot_delta(last_main, last_pcie, last_disk, step=step)
            last_main = counter_main
            last_pcie = counter_pcie
            last_disk = counter_disk
            socketio.emit("delta", delta)

    return socketio, app, push_loop


def main():
    args = parse_args()
    gpu_count = 0
    if HAS_NVML:
        gpu_count = pynvml.nvmlDeviceGetCount()

    dashboard_missing = _missing_dashboard_packages()
    dashboard_enabled = not dashboard_missing
    main_sec = args.main_interval / 1000.0
    disk_sec = args.disk_interval / 1000.0
    print(
        f"Main collect : {args.main_interval}ms (CPU + CPU procs + GPU mem/util/temp + network)"
    )
    print(
        f"PCIe collect : 10/s ({gpu_count} subprocess{'es' if gpu_count != 1 else ''})"
    )
    print(f"Disk collect : {args.disk_interval}ms (aggregate disk I/O)")
    if dashboard_enabled:
        print("Server mode  : shared endpoint (/ and /metrics)")
        print("Push interval: dynamic (75ms@1m .. 1000ms@15m)")
        print(
            f"Rolling window: {args.window}s ({args.window // 60}m {args.window % 60}s)"
        )
    else:
        print("Server mode  : Prometheus /metrics exporter")
        print(f"Dashboard    : disabled (missing: {', '.join(dashboard_missing)})")

    if HAS_NVML:
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_gib = mem.total / (1024**3)
            print(f"GPU {i}: {name} ({total_gib:.1f} GiB)")
            GPU_MEM_TOTAL_GIB.labels(gpu=str(i)).set(total_gib)
    else:
        print("No NVIDIA GPUs detected -- CPU + Disk only")
    print()

    collector = None
    updater = PrometheusUpdater(gpu_count)
    socketio = None
    app = None
    push_loop = None
    if dashboard_enabled:
        collector = MetricsCollector(
            window_sec=args.window,
            main_interval_ms=args.main_interval,
            disk_interval_ms=args.disk_interval,
            top_n=args.top_n,
        )
        collector.load_state()
        socketio, app, push_loop = build_monitor_server(collector)

    workers: list[multiprocessing.Process] = []
    pipes: list[multiprocessing.connection.Connection] = []

    # 1 PCIe subprocess per GPU (10/s)
    if HAS_NVML and gpu_count > 0:
        for gi in range(gpu_count):
            pcie_parent, pcie_child = multiprocessing.Pipe(duplex=False)
            pipes.append(pcie_parent)
            workers.append(
                multiprocessing.Process(
                    target=_pcie_worker, args=(pcie_child, gi), daemon=True
                )
            )

    # 1 main subprocess: CPU + GPU mem/util/temp + network (always started)
    main_parent, main_child = multiprocessing.Pipe(duplex=False)
    pipes.append(main_parent)
    workers.append(
        multiprocessing.Process(
            target=_main_worker,
            args=(main_child, gpu_count, main_sec),
            daemon=True,
        )
    )

    # 1 disk subprocess
    disk_parent, disk_child = multiprocessing.Pipe(duplex=False)
    pipes.append(disk_parent)
    workers.append(
        multiprocessing.Process(
            target=_disk_worker, args=(disk_child, disk_sec), daemon=True
        )
    )

    def _poll_pipes():
        live = list(pipes)
        while live:
            ready = multiprocessing.connection.wait(live, timeout=0.5)
            for conn in ready:
                try:
                    msg = conn.recv()
                except (EOFError, ConnectionResetError):
                    live.remove(conn)
                    print("[poll] worker pipe closed", flush=True)
                    continue
                tag = msg[0]
                if tag == "pcie":
                    updater.ingest_pcie(msg[2], msg[3], msg[4])
                    if collector is not None:
                        collector.ingest_pcie(msg[1], msg[2], msg[3], msg[4])
                elif tag == "main":
                    updater.ingest_main(msg[2], msg[3], msg[4], msg[5], msg[6])
                    if collector is not None:
                        collector.ingest_main(
                            msg[1], msg[2], msg[3], msg[4], msg[5], msg[6]
                        )
                elif tag == "disk":
                    updater.ingest_disk(msg[2], msg[3])
                    if collector is not None:
                        collector.ingest_disk(msg[1], msg[2], msg[3])

    for w in workers:
        w.start()
    threading.Thread(target=_poll_pipes, daemon=True).start()

    def _shutdown(signum, _frame):
        name = signal.Signals(signum).name
        if collector is not None:
            print(f"\n[{name}] saving metrics before exit.", flush=True)
            collector.save_state()
        else:
            print(f"\n[{name}] shutting down.", flush=True)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if dashboard_enabled:
        threading.Thread(target=push_loop, daemon=True).start()
        socketserver.TCPServer.allow_reuse_address = True
        print(f"Dashboard at http://{args.host}:{args.port}/")
        print(f"Prometheus metrics at http://{args.host}:{args.port}/metrics")
        socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)
    else:
        httpd = start_metrics_only_server(args.port, args.host, dashboard_missing)
        print(f"Prometheus metrics at http://{args.host}:{args.port}/metrics")
        try:
            httpd.serve_forever()
        except SystemExit:
            pass


if __name__ == "__main__":
    main()
