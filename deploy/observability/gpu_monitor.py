#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Real-time GPU/CPU/Disk/Network monitor.

Architecture (multiprocess, bypasses GIL):
  - 1 subprocess per GPU: PCIe TX/RX sampling at 10/s
  - 1 subprocess: CPU + GPU mem/util/temp + network at 5/s
  - 1 subprocess: aggregate disk I/O at 1/s
  - Main process: poll pipes, push deltas via WebSocket

Backend: Flask + flask-socketio. Frontend: Plotly.js + requestAnimationFrame.

Chart rows per GPU:
  1. GPU Memory by Process (GiB, stacked) + PCIe (GB/s, secondary y-axis)
  2. GPU Util (%) + Temperature (C, secondary y-axis)
Then:
  3. CPU Usage by Process Group (%, stacked top-5) + total CPU line
  4. Disk I/O (MB/s, aggregate) + Network (MB/s, secondary y-axis)

Usage:
    python3 gpu_monitor.py [--port 8051] [--host 0.0.0.0]
"""

import argparse
import json
import multiprocessing
import os
import signal
import socketserver
import threading
import time
from collections import deque
from pathlib import Path

import psutil
import pynvml
from flask import Flask, Response
from flask import request as flask_request
from flask_socketio import SocketIO

HAS_NVML = False
try:
    pynvml.nvmlInit()
    HAS_NVML = True
except pynvml.NVMLError as e:
    print(f"NVML init failed ({e}) -- GPU monitoring disabled")

PROCESS_COLORS = [
    "#00e5ff",  # cyan
    "#ff1744",  # red
    "#76ff03",  # green
    "#448aff",  # blue
    "#ff9100",  # orange
    "#d500f9",  # purple
    "#ffea00",  # yellow
    "#f50057",  # pink
    "#00e676",  # emerald
    "#ff6d00",  # dark orange
    "#18ffff",  # light cyan
    "#ea80fc",  # lavender
    "#b2ff59",  # lime
    "#ff80ab",  # rose
    "#64ffda",  # teal
    "#ffd740",  # amber
    "#e040fb",  # magenta
    "#ff9e80",  # salmon
    "#a7ffeb",  # mint
    "#ff8a80",  # coral
]
OTHER_COLOR = "#9e9e9e"
CACHE_DIR = Path.home() / ".cache" / "gpu_monitor"
CACHE_FILE = CACHE_DIR / "metrics.json"


def parse_args():
    p = argparse.ArgumentParser(description="Real-time GPU/CPU/Disk monitor")
    p.add_argument("--port", type=int, default=8051)
    p.add_argument("--host", default="127.0.0.1")
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
        help="Rolling window seconds (default 900 = 15min)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Show top N processes per chart, rest grouped as Other (gray)",
    )
    return p.parse_args()


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
            parts.append(f"PID={pid}")
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
            parts.append(f"PID={pid}")
            return ", ".join(parts)

        if cmdline and os.path.basename(cmdline[0]) == "node":
            label = _classify_node_process(full_cmd)
            if label:
                return f"{label}:{pid}"
        if cmdline and cmdline[0] == "docker" and "exec" in cmdline:
            if "node" in full_cmd:
                return f"Docker/Cursor:{pid}"

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
                    parts.append(f"PID={pid}")
                    return ", ".join(parts)
            if len(cmdline) > 1:
                script = os.path.basename(cmdline[1])
                if len(script) > 25:
                    script = script[:22] + "..."
                return f"{script}, PID={pid}"

        if cmdline:
            base = os.path.basename(cmdline[0])
            if len(base) > 25:
                base = base[:22] + "..."
            return f"{base}:{pid}"
        return f"{proc_name}:{pid}"
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
    """Extract the value after a CLI flag like --model. Also checks --model-path."""
    for f in (flag, flag + "-path"):
        try:
            idx = cmdline.index(f)
            if idx + 1 < len(cmdline):
                return cmdline[idx + 1]
        except ValueError:
            pass
    return ""


def _classify_node_process(full_cmd: str) -> str:
    if "--type=extensionHost" in full_cmd:
        return "ExtHost"
    if "cursorpyright" in full_cmd or "pyright" in full_cmd:
        return "Pylance"
    if "pylance" in full_cmd.lower():
        return "Pylance"
    if "--type=fileWatcher" in full_cmd:
        return "FileWatcher"
    if "--type=ptyHost" in full_cmd:
        return "PtyHost"
    if "server-main.js" in full_cmd:
        return "CursorServer"
    if "multiplex-server" in full_cmd:
        return "Multiplex"
    if "forwarder.js" in full_cmd:
        return "PortFwd"
    if "markdown-language-features" in full_cmd:
        return "Markdown"
    if "rust-analyzer" in full_cmd:
        return "RustAnalyzer"
    if "gitlens" in full_cmd.lower():
        return "GitLens"
    if "typescript" in full_cmd.lower():
        return "TSServer"
    if "eslint" in full_cmd.lower():
        return "ESLint"
    if "bootstrap-fork" in full_cmd:
        return "CursorWorker"
    if ".cursor-server" in full_cmd or ".vscode-server" in full_cmd:
        return "CursorNode"
    return ""


class ProcessTracker:
    def __init__(self, maxlen: int, prune: bool = True):
        self.maxlen = maxlen
        self._prune_enabled = prune
        self._len = 0
        self.series: dict[int, deque[float]] = {}
        self.names: dict[int, str] = {}
        self.first_seen: dict[int, float] = {}
        self._pid_slot: dict[int, int] = {}
        self._free_slots: list[int] = []
        self._next_slot: int = 0

    def new_pids(self, data: dict[int, float]) -> set[int]:
        """Return PIDs in data that aren't tracked yet (call outside lock)."""
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

    def load_dict(self, d: dict):
        self._len = d.get("_len", 0)
        for k, v in d.get("series", {}).items():
            pid = int(k)
            self.series[pid] = deque(v, maxlen=self.maxlen)
        self.names = {int(k): v for k, v in d.get("names", {}).items()}
        self.first_seen = {int(k): v for k, v in d.get("first_seen", {}).items()}
        self._pid_slot = {int(k): v for k, v in d.get("_pid_slot", {}).items()}
        self._free_slots = d.get("_free_slots", [])
        self._next_slot = d.get("_next_slot", 0)

    def _color_for(self, pid: int) -> str:
        slot = self._pid_slot.get(pid, 0)
        return PROCESS_COLORS[slot % len(PROCESS_COLORS)]

    def get_all_sorted(self) -> list[tuple[int, str, str, list[float]]]:
        """Return (pid, name, color, values) sorted by peak desc."""
        if not self.series:
            return []
        sorted_pids = sorted(
            self.series.keys(),
            key=lambda p: max(self.series[p]) if self.series[p] else 0,
            reverse=True,
        )
        return [
            (pid, self.names[pid], self._color_for(pid), list(self.series[pid]))
            for pid in sorted_pids
            if max(self.series[pid]) > 0
        ]

    def get_top_sorted(
        self, n=20, sort_by: str = "recency"
    ) -> list[tuple[int, str, str, list[float]]]:
        if not self.series:
            return []

        if sort_by == "recency":

            def _key(p):
                s = self.series[p]
                last_active = -1
                for i in range(len(s) - 1, -1, -1):
                    if s[i] > 0:
                        last_active = i
                        break
                return (last_active, max(s) if s else 0)

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
                for i, v in enumerate(self.series[pid]):
                    rest_vals[i] += v
                    if v > 0:
                        has_data = True
            if has_data:
                result.append((-1, "Other", OTHER_COLOR, rest_vals))
        return result


# ---------------------------------------------------------------------------
# Multiprocess workers (each gets its own GIL)
# ---------------------------------------------------------------------------


def _pcie_worker(conn, gpu_idx):
    """Subprocess: sample PCIe TX/RX for one GPU at ~10/s."""
    import pynvml as _nvml

    _nvml.nvmlInit()
    handle = _nvml.nvmlDeviceGetHandleByIndex(gpu_idx)
    TX = _nvml.NVML_PCIE_UTIL_TX_BYTES
    RX = _nvml.NVML_PCIE_UTIL_RX_BYTES
    while True:
        now = time.time()
        try:
            tx = _nvml.nvmlDeviceGetPcieThroughput(handle, TX) / (1024.0 * 1024.0)
            rx = _nvml.nvmlDeviceGetPcieThroughput(handle, RX) / (1024.0 * 1024.0)
        except _nvml.NVMLError:
            tx, rx = 0.0, 0.0
        conn.send(("pcie", now, gpu_idx, tx, rx))
        time.sleep(0.1)


def _main_worker(conn, gpu_count, interval_sec):
    """Subprocess: CPU (overall + per-name groups) + GPU mem/util + network at 5/s; temperature at 1/s."""
    import psutil as _ps
    import pynvml as _nvml

    _nvml.nvmlInit()
    handles = [_nvml.nvmlDeviceGetHandleByIndex(i) for i in range(gpu_count)]
    _ps.cpu_percent(interval=None)
    # Prime per-process cpu_percent so subsequent calls return real deltas
    list(_ps.process_iter(["name", "cpu_percent"]))
    ncpus = _ps.cpu_count() or 1
    prev_net = _ps.net_io_counters()
    last_mono = time.monotonic()
    # Temperature changes slowly; sample at ~1/s by collecting every temp_every-th cycle.
    temp_every = max(1, round(1.0 / (1.0 * interval_sec)))
    cycle = 0
    prev_temps: list[float] = [0.0] * gpu_count
    while True:
        now = time.time()
        mono = time.monotonic()
        dt = mono - last_mono
        if dt <= 0:
            dt = interval_sec
        cpu = _ps.cpu_percent(interval=None)
        # Per-process CPU grouped by name, normalised to total CPU capacity (0-100%)
        cpu_by_name: dict[str, float] = {}
        for p in _ps.process_iter(["name", "cpu_percent"]):
            try:
                name = p.info["name"] or "?"
                pct = p.info["cpu_percent"] or 0.0
                if pct > 0:
                    cpu_by_name[name] = cpu_by_name.get(name, 0.0) + pct
            except (_ps.NoSuchProcess, _ps.AccessDenied):
                continue
        for k in cpu_by_name:
            cpu_by_name[k] /= ncpus
        collect_temp = cycle % temp_every == 0
        gpu_data = []
        for gi, h in enumerate(handles):
            proc_mem: dict[int, float] = {}
            for getter in (
                _nvml.nvmlDeviceGetComputeRunningProcesses,
                _nvml.nvmlDeviceGetGraphicsRunningProcesses,
            ):
                try:
                    for p in getter(h):
                        mem_gib = (p.usedGpuMemory or 0) / (1024**3)
                        proc_mem[p.pid] = proc_mem.get(p.pid, 0.0) + mem_gib
                except _nvml.NVMLError:
                    pass
            try:
                util = _nvml.nvmlDeviceGetUtilizationRates(h).gpu
            except _nvml.NVMLError:
                util = 0.0
            if collect_temp:
                try:
                    prev_temps[gi] = float(
                        _nvml.nvmlDeviceGetTemperature(h, _nvml.NVML_TEMPERATURE_GPU)
                    )
                except _nvml.NVMLError:
                    pass
            gpu_data.append((proc_mem, util, prev_temps[gi]))
        net = _ps.net_io_counters()
        sent_rate = (net.bytes_sent - prev_net.bytes_sent) / (1024 * 1024) / dt
        recv_rate = (net.bytes_recv - prev_net.bytes_recv) / (1024 * 1024) / dt
        prev_net = net
        last_mono = mono
        conn.send(("main", now, cpu, gpu_data, sent_rate, recv_rate, cpu_by_name))
        cycle += 1
        time.sleep(interval_sec)


def _disk_worker(conn, interval_sec):
    """Subprocess: aggregate disk I/O (1/s)."""
    import psutil as _ps

    prev = _ps.disk_io_counters()
    last_mono = time.monotonic()
    while True:
        now = time.time()
        mono = time.monotonic()
        dt = mono - last_mono
        if dt <= 0:
            dt = interval_sec
        cur = _ps.disk_io_counters()
        read_mbps = (cur.read_bytes - prev.read_bytes) / (1024 * 1024) / dt
        write_mbps = (cur.write_bytes - prev.write_bytes) / (1024 * 1024) / dt
        prev = cur
        last_mono = mono
        conn.send(("disk", now, read_mbps, write_mbps))
        time.sleep(interval_sec)


class MetricsCollector:
    def __init__(
        self,
        window_sec: int,
        main_interval_ms: int = 200,
        disk_interval_ms: int = 500,
        pcie_rate_hz: int = 10,
        top_n: int = 10,
    ):
        self.lock = threading.Lock()
        maxlen_main = int(window_sec * 1000 / main_interval_ms)
        maxlen_pcie = window_sec * pcie_rate_hz
        maxlen_disk = int(window_sec * 1000 / disk_interval_ms)
        self.top_n = top_n
        # Main group: CPU + GPU mem/util/temp + network (5/s)
        self.ts_main: deque[float] = deque(maxlen=maxlen_main)
        self.counter_main: int = 0
        self.cpu_pct: deque[float] = deque(maxlen=maxlen_main)
        self.proc_gpu_mem: list[ProcessTracker] = []
        self.gpu_util: list[deque[float]] = []
        self.net_sent_mbps: deque[float] = deque(maxlen=maxlen_main)
        self.net_recv_mbps: deque[float] = deque(maxlen=maxlen_main)
        # PCIe group: per-GPU (max rate)
        self.ts_pcie: list[deque[float]] = []
        self.counter_pcie: list[int] = []
        self.gpu_pcie_tx: list[deque[float]] = []
        self.gpu_pcie_rx: list[deque[float]] = []
        self.has_pcie = False
        # CPU process groups (top-5 by name, same rate as main)
        self.proc_cpu = ProcessTracker(maxlen_main, prune=True)
        self._cpu_name_to_id: dict[str, int] = {}
        self._cpu_next_id: int = 1
        self._cpu_top_n: int = 5
        # Disk group: aggregate disk I/O (1/s)
        self.ts_disk: deque[float] = deque(maxlen=maxlen_disk)
        self.counter_disk: int = 0
        self.disk_read_mbps: deque[float] = deque(maxlen=maxlen_disk)
        self.disk_write_mbps: deque[float] = deque(maxlen=maxlen_disk)
        # GPU info
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
                self.gpu_pcie_tx.append(deque(maxlen=maxlen_pcie))
                self.gpu_pcie_rx.append(deque(maxlen=maxlen_pcie))
                self.ts_pcie.append(deque(maxlen=maxlen_pcie))
                self.counter_pcie.append(0)
        self.gpu_temp: list[deque[float]] = [
            deque(maxlen=maxlen_main) for _ in range(self.gpu_count)
        ]
        if self.gpu_count > 0:
            try:
                h = pynvml.nvmlDeviceGetHandleByIndex(0)
                pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES)
                self.has_pcie = True
            except pynvml.NVMLError:
                self.has_pcie = False

    def save_state(self):
        """Persist all time-series data to disk so a restart doesn't lose history."""
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
        """Restore time-series data from a previous run, if available."""
        if not CACHE_FILE.exists():
            return
        try:
            state = json.loads(CACHE_FILE.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"[load] failed to read cache: {e}", flush=True)
            return
        if state.get("gpu_count") != self.gpu_count:
            print(
                f"[load] GPU count mismatch (cached={state.get('gpu_count')}, "
                f"current={self.gpu_count}), ignoring cache",
                flush=True,
            )
            return
        age = time.time() - state.get("saved_at", 0)
        print(
            f"[load] restoring metrics from {CACHE_FILE} (saved {age:.0f}s ago)",
            flush=True,
        )
        with self.lock:
            ml_main = self.ts_main.maxlen
            ml_pcie = self.ts_pcie[0].maxlen if self.ts_pcie else 0
            ml_disk = self.ts_disk.maxlen
            self.ts_main = deque(state["ts_main"], maxlen=ml_main)
            self.counter_main = state["counter_main"]
            self.cpu_pct = deque(state["cpu_pct"], maxlen=ml_main)
            if "proc_cpu" in state:
                self.proc_cpu.load_dict(state["proc_cpu"])
                self._cpu_name_to_id = state.get("_cpu_name_to_id", {})
                self._cpu_next_id = state.get("_cpu_next_id", 1)
            self.net_sent_mbps = deque(state["net_sent_mbps"], maxlen=ml_main)
            self.net_recv_mbps = deque(state["net_recv_mbps"], maxlen=ml_main)
            for gi in range(self.gpu_count):
                self.ts_pcie[gi] = deque(state["ts_pcie"][gi], maxlen=ml_pcie)
                self.counter_pcie[gi] = state["counter_pcie"][gi]
                self.gpu_pcie_tx[gi] = deque(state["gpu_pcie_tx"][gi], maxlen=ml_pcie)
                self.gpu_pcie_rx[gi] = deque(state["gpu_pcie_rx"][gi], maxlen=ml_pcie)
                self.gpu_util[gi] = deque(state["gpu_util"][gi], maxlen=ml_main)
                self.gpu_temp[gi] = deque(state["gpu_temp"][gi], maxlen=ml_main)
                self.proc_gpu_mem[gi].load_dict(state["proc_gpu_mem"][gi])
            self.ts_disk = deque(state["ts_disk"], maxlen=ml_disk)
            self.counter_disk = state["counter_disk"]
            if "disk_read_mbps" in state:
                self.disk_read_mbps = deque(state["disk_read_mbps"], maxlen=ml_disk)
                self.disk_write_mbps = deque(state["disk_write_mbps"], maxlen=ml_disk)
        print(
            f"[load] restored {len(self.ts_main)} main samples, "
            f"{len(self.ts_disk)} disk samples",
            flush=True,
        )

    def ingest_pcie(self, now: float, gpu_idx: int, tx_val: float, rx_val: float):
        """Store PCIe data for one GPU received from its worker process."""
        with self.lock:
            self.ts_pcie[gpu_idx].append(now)
            self.counter_pcie[gpu_idx] += 1
            self.gpu_pcie_tx[gpu_idx].append(tx_val)
            self.gpu_pcie_rx[gpu_idx].append(rx_val)

    def _cpu_name_id(self, name: str) -> int:
        """Stable synthetic ID for a process-name group."""
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
        """Store CPU + GPU mem/util/temp + network (main group, 5/s)."""
        gpu_names: list[dict[int, str]] = []
        for i, (gpu_proc_mem, _, _) in enumerate(gpu_data):
            new = self.proc_gpu_mem[i].new_pids(gpu_proc_mem)
            gpu_names.append({pid: _resolve_process_name(pid) for pid in new})
        # Convert cpu_by_name (str->float) to synthetic-id->float for ProcessTracker
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
        """Store aggregate disk I/O (1/s)."""
        with self.lock:
            self.ts_disk.append(now)
            self.counter_disk += 1
            self.disk_read_mbps.append(read_mbps)
            self.disk_write_mbps.append(write_mbps)

    @staticmethod
    def _downsample_init(times, arrays, max_recent=1000, max_total=1500):
        """Downsample for init: keep last max_recent at full res, older data subsampled."""
        n = len(times)
        if n <= max_total:
            return times, arrays
        older_count = n - max_recent
        older_target = max_total - max_recent
        step = max(1, older_count // older_target)
        # Build indices: subsampled older + full recent
        older_indices = list(range(0, older_count, step))
        recent_indices = list(range(older_count, n))
        indices = older_indices + recent_indices
        ds_times = [times[i] for i in indices]
        ds_arrays = [
            [arr[i] for i in indices] if len(arr) == n else arr for arr in arrays
        ]
        return ds_times, ds_arrays

    def snapshot_full(self) -> dict:
        """Return full state for initial browser load (downsampled for browser perf)."""
        with self.lock:
            # --- Main group: CPU + CPU procs + GPU mem/util/temp + network ---
            main_vals = [list(self.cpu_pct)]
            cpu_procs_full = self.proc_cpu.get_top_sorted(
                self._cpu_top_n, sort_by="total"
            )
            for _, _, _, v in cpu_procs_full:
                main_vals.append(v)
            gpu_mem_per_gpu_full = []
            for gi in range(self.gpu_count):
                procs = self.proc_gpu_mem[gi].get_top_sorted(self.top_n)
                gpu_mem_per_gpu_full.append(procs)
                for _, _, _, v in procs:
                    main_vals.append(v)
            gpu_util = [list(self.gpu_util[i]) for i in range(self.gpu_count)]
            gpu_temp = [list(self.gpu_temp[i]) for i in range(self.gpu_count)]
            main_vals.extend(gpu_util)
            main_vals.extend(gpu_temp)
            main_vals.append(list(self.net_sent_mbps))
            main_vals.append(list(self.net_recv_mbps))
            ts_main_ds, main_ds = self._downsample_init(list(self.ts_main), main_vals)
            midx = 0
            ds_cpu = main_ds[midx]
            midx += 1
            cpu_procs = []
            for p, n, c, v in cpu_procs_full:
                cpu_procs.append(
                    {"id": p, "name": n, "color": c, "vals": main_ds[midx]}
                )
                midx += 1
            gpu_mem_per_gpu = []
            for gi in range(self.gpu_count):
                ds_procs = []
                for p, n, c, v in gpu_mem_per_gpu_full[gi]:
                    ds_procs.append(
                        {
                            "pid": p,
                            "name": n,
                            "color": c,
                            "vals": main_ds[midx],
                            "first_seen": self.proc_gpu_mem[gi].first_seen.get(p, 0),
                        }
                    )
                    midx += 1
                gpu_mem_per_gpu.append(ds_procs)
            ds_gpu_util = [main_ds[midx + i] for i in range(self.gpu_count)]
            midx += self.gpu_count
            ds_gpu_temp = [main_ds[midx + i] for i in range(self.gpu_count)]
            midx += self.gpu_count
            ds_net_sent = main_ds[midx]
            ds_net_recv = main_ds[midx + 1]

            # --- PCIe group (per-GPU timestamps) ---
            ds_pcie_tx = []
            ds_pcie_rx = []
            ts_pcie_per_gpu = []
            if self.has_pcie:
                for i in range(self.gpu_count):
                    ts_raw = list(self.ts_pcie[i])
                    tx_raw = list(self.gpu_pcie_tx[i])
                    rx_raw = list(self.gpu_pcie_rx[i])
                    ts_ds, vals_ds = self._downsample_init(ts_raw, [tx_raw, rx_raw])
                    ts_pcie_per_gpu.append(ts_ds)
                    ds_pcie_tx.append(vals_ds[0])
                    ds_pcie_rx.append(vals_ds[1])

            # --- Disk group (aggregate) ---
            disk_vals = [list(self.disk_read_mbps), list(self.disk_write_mbps)]
            ts_disk_ds, disk_ds = self._downsample_init(list(self.ts_disk), disk_vals)
            ds_disk_read = disk_ds[0]
            ds_disk_write = disk_ds[1]

            # Build compact GPU summary like "[2x RTX 6000 Ada]"
            gpu_summary = ""
            if self.gpu_names:
                from collections import Counter

                counts = Counter(self.gpu_names)
                parts = []
                for name, cnt in counts.items():
                    short = (
                        name.replace("NVIDIA ", "").replace("Generation", "").strip()
                    )
                    parts.append(f"{cnt}x {short}" if cnt > 1 else short)
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
        """Return new data since the given counters, subsampled by step."""
        with self.lock:
            # Main group
            new_main = self.counter_main - since_main
            main_len = len(self.ts_main)
            main_idx = max(0, main_len - new_main) if new_main > 0 else main_len
            sl_m = slice(main_idx, None, step)
            new_ts_main = list(self.ts_main)[sl_m]
            cpu = list(self.cpu_pct)[sl_m]
            cpu_procs = self.proc_cpu.get_top_sorted(self._cpu_top_n, sort_by="total")
            cpu_procs_delta = [
                {
                    "id": p,
                    "name": n,
                    "color": c,
                    "vals": v[sl_m] if main_idx < len(v) else [],
                }
                for p, n, c, v in cpu_procs
            ]
            cpu_proc_keys = [p for p, _, _, _ in cpu_procs]
            gpu_mem_per_gpu = []
            gpu_mem_keys_per_gpu = []
            for gi in range(self.gpu_count):
                gm = self.proc_gpu_mem[gi].get_top_sorted(self.top_n)
                gpu_mem_per_gpu.append(
                    [
                        {
                            "pid": p,
                            "name": n,
                            "color": c,
                            "vals": v[sl_m] if main_idx < len(v) else [],
                        }
                        for p, n, c, v in gm
                    ]
                )
                gpu_mem_keys_per_gpu.append([p for p, _, _, _ in gm])
            gpu_util = [list(self.gpu_util[i])[sl_m] for i in range(self.gpu_count)]
            gpu_temp = [list(self.gpu_temp[i])[sl_m] for i in range(self.gpu_count)]
            net_sent = list(self.net_sent_mbps)[sl_m]
            net_recv = list(self.net_recv_mbps)[sl_m]

            # PCIe group (per-GPU timestamps, thinned to ~5/s)
            pcie_step = max(step, 2)
            pcie_tx = []
            pcie_rx = []
            new_ts_pcie = []
            if self.has_pcie:
                for i in range(self.gpu_count):
                    sp = since_pcie[i] if i < len(since_pcie) else 0
                    new_p = self.counter_pcie[i] - sp
                    p_len = len(self.ts_pcie[i])
                    p_idx = max(0, p_len - new_p) if new_p > 0 else p_len
                    sl_p = slice(p_idx, None, pcie_step)
                    new_ts_pcie.append(list(self.ts_pcie[i])[sl_p])
                    pcie_tx.append(list(self.gpu_pcie_tx[i])[sl_p])
                    pcie_rx.append(list(self.gpu_pcie_rx[i])[sl_p])

            # Disk group (aggregate)
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
# HTML / JS Frontend (embedded)
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>DynamoGPUMonitor</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d0d14; color: #e0e0e0; font-family: monospace; overflow: hidden; }
  #toolbar { display: flex; justify-content: space-between; align-items: center; padding: 12px 20px; }
  #toolbar h1 { font-size: 22px; }
  .btn { border: none; padding: 6px 14px; cursor: pointer; font-family: monospace;
         font-weight: bold; margin-left: 4px; border-radius: 4px; font-size: 13px; }
  .range-btn { background: #2a2a3c; color: #e0e0e0; border: 1px solid #444; }
  .range-btn.active { background: #444466; }
  #pause-btn { background: #ffcc00; color: #0d0d14; margin-left: 12px; }
  #snapshot-btn { background: #00e676; color: #0d0d14; }
  #chart { width: 100%; height: calc(100vh - 50px); }
  #status { position: fixed; bottom: 8px; left: 20px; font-size: 11px; color: #666; }
</style>
</head>
<body>
<div id="toolbar">
  <h1>DynamoGPUMonitor</h1>
  <div>
    <button class="btn range-btn" data-min="1">1m</button>
    <button class="btn range-btn active" data-min="2">2m</button>
    <button class="btn range-btn" data-min="5">5m</button>
    <button class="btn range-btn" data-min="10">10m</button>
    <button class="btn range-btn" data-min="15">15m</button>
    <button class="btn" id="pause-btn">Pause</button>
    <button class="btn" id="snapshot-btn">Snapshot</button>
  </div>
</div>
<div id="chart"></div>
<div id="status">Connecting...</div>

<script>
(function() {
  "use strict";

  const MAX_POINTS = 3000;
  let viewMinutes = 2;
  let paused = false;
  let syncing = false;
  let gd = null;
  let config = null;
  let traceMap = {};
  let traceCount = 0;
  let prev_srv_main = 0, prev_srv_pcie = [], prev_srv_disk = 0;
  let cur_srv_main = 0, cur_srv_pcie = [], cur_srv_disk = 0;
  let lastRateCheck = Date.now();
  let rate_main = 0, rate_pcie = 0, rate_disk = 0;
  let rate_pcie_per_gpu = [];
  let rowAnnotations = []; // [{index, baseText, group}]
  let pendingDeltas = [];
  let rafScheduled = false;

  function flushDeltas() {
    rafScheduled = false;
    if (!gd || !config || pendingDeltas.length === 0) return;
    const batch = pendingDeltas;
    pendingDeltas = [];
    const gc = config.gpu_count || 0;
    const nr = (gc > 0 ? gc * 2 : 0) + 2;

    // Accumulators: one extendTraces call per group
    const mx = {}, my = {};  // traceIndex -> [concatenated x], [concatenated y]
    const pcieX = {}, pcieY = {};
    const diskX = {}, diskY = {};

    for (const msg of batch) {
      const tm = msg.ts_main || [], tp = msg.ts_pcie || [], td = msg.ts_disk || [];

      // Main group
      if (tm.length > 0) {
        const times_m = tm.map(t => new Date(t * 1000));
        // CPU process groups
        for (let i = 0; i < (msg.cpu_procs || []).length; i++) {
          const idx = traceMap['cpu_proc_' + msg.cpu_procs[i].id];
          if (idx !== undefined) {
            if (!mx[idx]) { mx[idx] = []; my[idx] = []; }
            mx[idx].push(...times_m); my[idx].push(...msg.cpu_procs[i].vals);
          }
        }
        // CPU total line
        const cpuIdx = traceMap['cpu'];
        if (cpuIdx !== undefined) {
          if (!mx[cpuIdx]) { mx[cpuIdx] = []; my[cpuIdx] = []; }
          mx[cpuIdx].push(...times_m); my[cpuIdx].push(...msg.cpu);
        }
        for (let gi = 0; gi < (msg.gpu_mem || []).length; gi++) {
          for (let i = 0; i < msg.gpu_mem[gi].length; i++) {
            const idx = traceMap['gpu_mem_' + gi + '_' + msg.gpu_mem[gi][i].pid];
            if (idx !== undefined) {
              if (!mx[idx]) { mx[idx] = []; my[idx] = []; }
              mx[idx].push(...times_m); my[idx].push(...msg.gpu_mem[gi][i].vals);
            }
          }
        }
        for (let i = 0; i < (msg.gpu_util || []).length; i++) {
          const idx = traceMap['gpu_util_' + i];
          if (idx !== undefined) {
            if (!mx[idx]) { mx[idx] = []; my[idx] = []; }
            mx[idx].push(...times_m); my[idx].push(...msg.gpu_util[i]);
          }
        }
        for (let i = 0; i < (msg.gpu_temp || []).length; i++) {
          const idx = traceMap['gpu_temp_' + i];
          if (idx !== undefined) {
            if (!mx[idx]) { mx[idx] = []; my[idx] = []; }
            mx[idx].push(...times_m); my[idx].push(...msg.gpu_temp[i]);
          }
        }
        if (traceMap['net_sent'] !== undefined) {
          const si = traceMap['net_sent'], ri = traceMap['net_recv'];
          if (!mx[si]) { mx[si] = []; my[si] = []; }
          if (!mx[ri]) { mx[ri] = []; my[ri] = []; }
          mx[si].push(...times_m); my[si].push(...msg.net_sent);
          mx[ri].push(...times_m); my[ri].push(...msg.net_recv);
        }
      }

      // PCIe group
      if (tp.length > 0 && config.has_pcie) {
        for (let i = 0; i < (msg.pcie_tx || []).length; i++) {
          if (!tp[i] || tp[i].length === 0) continue;
          const times_p = tp[i].map(t => new Date(t * 1000));
          const txIdx = traceMap['pcie_tx_' + i], rxIdx = traceMap['pcie_rx_' + i];
          if (txIdx !== undefined) {
            if (!pcieX[txIdx]) { pcieX[txIdx] = []; pcieY[txIdx] = []; }
            pcieX[txIdx].push(...times_p); pcieY[txIdx].push(...msg.pcie_tx[i]);
          }
          if (rxIdx !== undefined) {
            if (!pcieX[rxIdx]) { pcieX[rxIdx] = []; pcieY[rxIdx] = []; }
            pcieX[rxIdx].push(...times_p); pcieY[rxIdx].push(...msg.pcie_rx[i]);
          }
        }
      }

      // Disk group (aggregate)
      if (td.length > 0) {
        const times_d = td.map(t => new Date(t * 1000));
        const dtIdx = traceMap['disk_total'];
        if (dtIdx !== undefined) {
          if (!diskX[dtIdx]) { diskX[dtIdx] = []; diskY[dtIdx] = []; }
          const dr = msg.disk_read || [], dw = msg.disk_write || [];
          for (let k = 0; k < dr.length; k++) {
            diskX[dtIdx].push(times_d[k]);
            diskY[dtIdx].push(dr[k] + (dw[k] || 0));
          }
        }
      }
    }

    // Single extendTraces per group
    const mIndices = Object.keys(mx).map(Number);
    if (mIndices.length > 0) {
      Plotly.extendTraces(gd, {
        x: mIndices.map(i => mx[i]),
        y: mIndices.map(i => my[i])
      }, mIndices, MAX_POINTS);
    }
    const pIndices = Object.keys(pcieX).map(Number);
    if (pIndices.length > 0) {
      Plotly.extendTraces(gd, {
        x: pIndices.map(i => pcieX[i]),
        y: pIndices.map(i => pcieY[i])
      }, pIndices, MAX_POINTS);
    }
    const dIndices = Object.keys(diskX).map(Number);
    if (dIndices.length > 0) {
      Plotly.extendTraces(gd, {
        x: dIndices.map(i => diskX[i]),
        y: dIndices.map(i => diskY[i])
      }, dIndices, MAX_POINTS);
    }

    // Single scroll + annotation update
    if (!paused && viewMinutes > 0) {
      const now = new Date();
      const start = new Date(now.getTime() - viewMinutes * 60000);
      for (let r = 1; r <= nr; r++) {
        const ax = r === 1 ? 'xaxis' : 'xaxis' + r;
        gd.layout[ax].range = [start, now];
        gd.layout[ax].autorange = false;
      }
    }
  }

  const socket = io({transports: ['websocket', 'polling'], reconnection: true, reconnectionDelay: 500, reconnectionAttempts: Infinity, timeout: 60000});

  socket.on('connect', function() {
    document.getElementById('status').textContent = 'Connected. Waiting for data...';
    socket.emit('view_minutes', viewMinutes);
    if (config) {
      socket.emit('request_init');
    } else {
      setTimeout(function() {
        if (!config) socket.emit('request_init');
      }, 3000);
    }
  });

  socket.on('disconnect', function() {
    document.getElementById('status').textContent = 'Disconnected -- reconnecting...';
  });

  socket.on('init', function(msg) {
    var savedRange = null;
    var savedVM = viewMinutes;
    if (gd && viewMinutes === 0) {
      var ax1 = gd.layout.xaxis;
      if (ax1 && ax1.range) savedRange = [ax1.range[0], ax1.range[1]];
    }
    config = msg;
    pendingDeltas = [];
    rafScheduled = false;
    cur_srv_main = msg.counter_main || 0;
    prev_srv_main = cur_srv_main;
    cur_srv_pcie = msg.counter_pcie || [];
    prev_srv_pcie = cur_srv_pcie.slice();
    cur_srv_disk = msg.counter_disk || 0;
    prev_srv_disk = cur_srv_disk;
    lastRateCheck = Date.now();
    buildChart(msg);
    if (savedRange && savedVM === 0) {
      viewMinutes = 0;
      var gc = config.gpu_count || 0;
      var nr = (gc > 0 ? gc * 2 : 0) + 2;
      var upd = {};
      for (var r = 1; r <= nr; r++) {
        var ax = r === 1 ? 'xaxis' : 'xaxis' + r;
        upd[ax + '.range'] = savedRange;
        upd[ax + '.autorange'] = false;
      }
      syncing = true;
      Plotly.relayout(gd, upd).then(function() { syncing = false; });
    }
    if (msg.hostname) {
      const gpuTag = msg.gpu_summary ? ' ' + msg.gpu_summary : '';
      const title = 'DynamoGPUMonitor - ' + msg.hostname + gpuTag;
      document.querySelector('#toolbar h1').textContent = title;
      document.title = title;
    }
    if (viewMinutes > 0) {
      document.getElementById('status').textContent = 'Live';
    }
    startScrollLoop();
  });

  socket.on('delta', function(msg) {
    if (!gd || !config) return;
    const tm = msg.ts_main || [], tp = msg.ts_pcie || [], td = msg.ts_disk || [];
    const hasPcieData = tp.length > 0 && tp.some(a => a.length > 0);
    if (tm.length === 0 && !hasPcieData && td.length === 0) return;

    cur_srv_main = msg.counter_main || cur_srv_main;
    if (Array.isArray(msg.counter_pcie)) cur_srv_pcie = msg.counter_pcie;
    cur_srv_disk = msg.counter_disk || cur_srv_disk;

    function sameKeySet(a, b) {
      if (!a || !b || a.length !== b.length) return false;
      const sa = JSON.stringify(a.map(x => Array.isArray(x) ? x.slice().sort() : x));
      const sb = JSON.stringify(b.map(x => Array.isArray(x) ? x.slice().sort() : x));
      return sa === sb;
    }
    if (!sameKeySet(msg.gpu_mem_keys, config._last_gpu_keys) ||
        !sameKeySet([msg.cpu_proc_keys || []], [config._last_cpu_keys || []])) {
      socket.emit('request_init');
      return;
    }

    pendingDeltas.push(msg);
    if (!rafScheduled) {
      rafScheduled = true;
      requestAnimationFrame(flushDeltas);
    }
  });


  function buildChart(msg) {
    gd = document.getElementById('chart');
    traceMap = {};
    traceCount = 0;
    rowAnnotations = [];
    const traces = [];
    const times_main = (msg.ts_main || []).map(t => new Date(t * 1000));
    const ts_pcie_raw = msg.ts_pcie || [];
    const times_pcie_per_gpu = ts_pcie_raw.map(a => (a || []).map(t => new Date(t * 1000)));
    const times_disk = (msg.ts_disk || []).map(t => new Date(t * 1000));

    const hasGpu = msg.gpu_count > 0;
    const hasPcie = msg.has_pcie;
    const gpuCount = msg.gpu_count;
    const sysLegName = hasGpu ? (gpuCount === 1 ? 'legend2' : 'legend' + (gpuCount + 1)) : 'legend';
    // Rows: 2*N (mem+util per GPU) + 1 cpu + 1 disk+net
    const nRows = (hasGpu ? gpuCount * 2 : 0) + 2;
    // Row weights: GPU mem=5, GPU util=5, CPU=3, Disk+Net=3
    const rowWeights = [];
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        rowWeights.push(7); // mem
        rowWeights.push(3); // util+temp
      }
    }
    rowWeights.push(3); // CPU
    rowWeights.push(3); // Disk+Net

    const xax = r => r === 1 ? 'x' : 'x' + r;
    const yax = r => r === 1 ? 'y' : 'y' + r;
    const pcieYIdxPerGpu = []; // secondary y-axis index per GPU

    let row = 0;

    const gpuUtilColors = ['#76b900', '#ff1744', '#448aff', '#ffea00', '#d500f9', '#ff9100', '#00e676', '#ff6d00'];
    const gpuPcieColors = ['#40c4ff', '#ff6e40', '#b388ff', '#ffd54f', '#ea80fc', '#ffab40', '#64ffda', '#ff8a65'];

    // Per-GPU blocks: Memory+PCIe row + Util+Temp row
    const pcieYIdxPerGpu_mem = [];
    const tempYIdxPerGpu = [];
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        // Memory + PCIe row
        row++;
        const memRow = row;
        const gpuMem = msg.gpu_mem[gi] || [];
        const stackName = 'gpu_mem_' + gi;
        const gpuLeg = gi === 0 ? 'legend' : 'legend' + (gi + 1);
        const memPatterns = ['/', '\\', 'x', '+', '-', '|', '.'];
        const memColors = gpuMem.map(m => m.color);
        const similarSet = findSimilarColors(memColors, 30);
        for (let i = gpuMem.length - 1; i >= 0; i--) {
          const m = gpuMem[i];
          traceMap['gpu_mem_' + gi + '_' + m.pid] = traceCount++;
          const trace = {
            x: times_main, y: m.vals, name: m.name, type: 'scatter', mode: 'none',
            line: {width: 0, color: m.color}, stackgroup: stackName,
            fillcolor: hexToRgba(m.color, 0.6),
            xaxis: xax(memRow), yaxis: yax(memRow),
            legendgroup: 'gpu_mem_' + gi, legendgrouptitle: {text: 'GPU ' + gi + ' Mem'}, legend: gpuLeg,
          };
          if (similarSet.has(i)) {
            const pat = memPatterns[Math.abs(m.pid) % memPatterns.length];
            trace.fillpattern = {shape: pat, solidity: 0.15, size: 8, bgcolor: hexToRgba(m.color, 0.6), fgcolor: 'rgba(13,13,20,0.4)'};
          }
          traces.push(trace);
        }
        if (gpuMem.length === 0) {
          traceMap['gpu_mem_' + gi + '_empty'] = traceCount++;
          traces.push({
            x: times_main, y: times_main.map(() => 0), name: 'GPU ' + gi + ' Mem: 0', type: 'scatter',
            mode: 'lines', line: {width: 1, color: '#555'},
            xaxis: xax(memRow), yaxis: yax(memRow), legendgroup: 'gpu_mem_' + gi, legend: gpuLeg,
          });
        }
        if (hasPcie) {
          const pyIdx = nRows + 1 + gi;
          pcieYIdxPerGpu_mem.push({idx: pyIdx, memRow: memRow});
          const pcieY = 'y' + pyIdx;
          const tp_gpu = times_pcie_per_gpu[gi] || [];
          traceMap['pcie_tx_' + gi] = traceCount++;
          traces.push({
            x: tp_gpu, y: msg.pcie_tx[gi], name: 'PCIe W (GPU ' + gi + ')', type: 'scatter',
            line: {color: '#ef5350', width: 0.5},
            xaxis: xax(memRow), yaxis: pcieY,
            legendgroup: 'pcie_' + gi, legendgrouptitle: {text: 'GPU ' + gi + ' PCIe'}, legend: gpuLeg,
          });
          traceMap['pcie_rx_' + gi] = traceCount++;
          traces.push({
            x: tp_gpu, y: msg.pcie_rx[gi], name: 'PCIe R (GPU ' + gi + ')', type: 'scatter',
            line: {color: '#69f0ae', width: 0.5, dash: 'dot'},
            xaxis: xax(memRow), yaxis: pcieY,
            legendgroup: 'pcie_' + gi, legend: gpuLeg,
          });
        }

        // Util + Temperature row for this GPU
        row++;
        const utilRow = row;
        const uColor = '#76b900';
        traceMap['gpu_util_' + gi] = traceCount++;
        traces.push({
          x: times_main, y: msg.gpu_util[gi], name: 'GPU ' + gi + ' Util', type: 'scatter',
          line: {color: uColor, width: 2}, fill: 'tozeroy',
          fillcolor: hexToRgba(uColor, 0.65),
          xaxis: xax(utilRow), yaxis: yax(utilRow),
          legendgroup: 'util_' + gi, legendgrouptitle: {text: 'GPU ' + gi + ' Util'}, legend: gpuLeg,
        });
        // GPU Temperature on secondary y-axis
        const tIdx = nRows + gpuCount + 1 + gi;
        tempYIdxPerGpu.push({idx: tIdx, utilRow: utilRow, gi: gi});
        const tempY = 'y' + tIdx;
        const tColor = '#ff6e40';
        traceMap['gpu_temp_' + gi] = traceCount++;
        traces.push({
          x: times_main, y: msg.gpu_temp[gi], name: 'GPU ' + gi + ' Temp', type: 'scatter',
          line: {color: tColor, width: 1.5, dash: 'dot'},
          xaxis: xax(utilRow), yaxis: tempY,
          legendgroup: 'util_' + gi, legend: gpuLeg,
        });
      }
    }

    // CPU row: overall line + stacked process groups
    row++;
    const cpuRow = row;
    const cpuProcs = msg.cpu_procs || [];
    // Stacked areas for process groups (bottom-to-top)
    for (let i = cpuProcs.length - 1; i >= 0; i--) {
      const cp = cpuProcs[i];
      traceMap['cpu_proc_' + cp.id] = traceCount++;
      traces.push({
        x: times_main, y: cp.vals, name: cp.name, type: 'scatter', mode: 'none',
        line: {width: 0, color: cp.color}, stackgroup: 'cpu_procs',
        fillcolor: hexToRgba(cp.color, 0.55),
        xaxis: xax(cpuRow), yaxis: yax(cpuRow),
        legendgroup: 'cpu', legendgrouptitle: {text: 'CPU'}, legend: sysLegName,
      });
    }
    // Overall CPU as thin dotted reference line
    traceMap['cpu'] = traceCount++;
    traces.push({
      x: times_main, y: msg.cpu, name: 'CPU total', type: 'scatter',
      line: {color: '#ffcc00', width: 1.5, dash: 'dot'},
      xaxis: xax(cpuRow), yaxis: yax(cpuRow),
      legendgroup: 'cpu', legend: sysLegName,
    });

    // Disk I/O + Network I/O row
    row++;
    const diskRow = row;
    const netYIdx = nRows + pcieYIdxPerGpu_mem.length + tempYIdxPerGpu.length + 1;
    const netY = 'y' + netYIdx;
    // Combined disk I/O (read + write) as single filled area
    const diskTotal = (msg.disk_read || []).map((v, i) => v + (msg.disk_write || [])[i]);
    traceMap['disk_total'] = traceCount++;
    traces.push({
      x: times_disk, y: diskTotal, name: 'Disk I/O', type: 'scatter',
      line: {color: '#ff9800', width: 1, shape: 'spline', smoothing: 0.8},
      fill: 'tozeroy', fillcolor: hexToRgba('#ff9800', 0.35),
      xaxis: xax(diskRow), yaxis: yax(diskRow),
      legendgroup: 'disk_io', legendgrouptitle: {text: 'Disk I/O'}, legend: sysLegName,
    });
    // Network I/O overlaid
    traceMap['net_sent'] = traceCount++;
    traces.push({
      x: times_main, y: msg.net_sent, name: 'Net TX', type: 'scatter',
      line: {color: '#18ffff', width: 0.5, shape: 'spline', smoothing: 1.3},
      xaxis: xax(diskRow), yaxis: netY,
      legendgroup: 'net', legendgrouptitle: {text: 'Network'}, legend: sysLegName,
    });
    traceMap['net_recv'] = traceCount++;
    traces.push({
      x: times_main, y: msg.net_recv, name: 'Net RX', type: 'scatter',
      line: {color: '#ff80ab', width: 0.5, shape: 'spline', smoothing: 1.3},
      xaxis: xax(diskRow), yaxis: netY,
      legendgroup: 'net', legend: sysLegName,
    });

    // Store keys for change detection
    config._last_gpu_keys = msg.gpu_mem.map(gm => gm.map(m => m.pid));
    config._last_cpu_keys = (msg.cpu_procs || []).map(c => c.id);

    const bgColors = ['#101020', '#181830'];
    const rowLabels = [];
    const rowGroups = [];
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        rowLabels.push(hasPcie ? 'GPU ' + gi + ' Memory (GiB) + PCIe (GB/s)' : 'GPU ' + gi + ' Memory (GiB)');
        rowGroups.push(hasPcie ? 'main_pcie' : 'main');
        rowLabels.push('GPU ' + gi + ' %-util + GPU \u00b0C');
        rowGroups.push('main');
      }
    }
    rowLabels.push('CPU Usage (%)');
    rowGroups.push('main');
    rowLabels.push('Disk I/O (MB/s) + Network (MB/s)');
    rowGroups.push('disk');

    // Per-row gaps: small between mem+util (same GPU), large between GPU groups and before CPU
    const rowGaps = [];
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        if (gi > 0) rowGaps.push(0.04);  // gap before this GPU group's mem row
        else rowGaps.push(0);             // no gap before first row
        rowGaps.push(0.002);              // tiny gap between mem and util (same GPU)
      }
    }
    rowGaps.push(hasGpu ? 0.04 : 0);     // gap before CPU
    rowGaps.push(0.02);                   // gap before Disk+Net

    const domain = function(i, total) {
      const totalWeight = rowWeights.reduce((a, b) => a + b, 0);
      const totalGap = rowGaps.reduce((a, b) => a + b, 0);
      const usable = 1 - totalGap;
      let y1 = 1;
      for (let r = 0; r < i; r++) {
        y1 -= (rowWeights[r] / totalWeight) * usable + rowGaps[r + 1];
      }
      const h = (rowWeights[i] / totalWeight) * usable;
      return [y1 - h, y1];
    };

    const layout = {
      paper_bgcolor: '#0d0d14', plot_bgcolor: '#141420',
      font: {family: 'monospace', color: '#e0e0e0', size: 11},
      margin: {l: 70, r: 20, t: 10, b: 35},
      showlegend: true,
      hovermode: false,
      shapes: [], annotations: [],
    };

    // Build axis layout dynamically for all rows
    for (let r = 1; r <= nRows; r++) {
      const xName = r === 1 ? 'xaxis' : 'xaxis' + r;
      const yName = r === 1 ? 'yaxis' : 'yaxis' + r;
      const yRef = r === 1 ? 'y' : 'y' + r;
      const showTick = (r === nRows);
      const xRange = viewMinutes > 0 ? [new Date(Date.now() - viewMinutes * 60000), new Date()] : undefined;
      layout[xName] = {domain: [0, 0.92], anchor: yRef, showticklabels: showTick, gridcolor: '#262640', range: xRange};
      layout[yName] = {domain: domain(r - 1, nRows), gridcolor: '#262640'};
    }

    // Set y-axis types per row
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        const memRowIdx = 2 * gi + 1;
        const utilRowIdx = 2 * gi + 2;
        const memYName = memRowIdx === 1 ? 'yaxis' : 'yaxis' + memRowIdx;
        const utilYName = 'yaxis' + utilRowIdx;
        const maxGib = msg.gpu_mem_total_gib[gi] || 1;
        layout[memYName].title = {text: '<span style="color:#76b900">GPU ' + gi + ' GiB</span>', font: {size: 10}};
        layout[memYName].rangemode = 'tozero';
        layout[memYName].autorange = true;
        layout[utilYName].title = {text: '<span style="color:#76b900">GPU ' + gi + ' %-util</span>', font: {size: 10}};
        layout[utilYName].range = [0, 105];
      }
      // PCIe secondary axes (one per GPU mem row)
      for (const pci of pcieYIdxPerGpu_mem) {
        const pcieYName = 'yaxis' + pci.idx;
        layout[pcieYName] = {
          domain: domain(pci.memRow - 1, nRows),
          title: {text: 'PCIe GB/s (log)', font: {color: '#999', size: 10}},
          overlaying: yax(pci.memRow), side: 'right', showgrid: false,
          type: 'log', autorange: true,
        };
      }
      // GPU Temp secondary axes (one per GPU util row)
      for (const tmp of tempYIdxPerGpu) {
        const tempYName = 'yaxis' + tmp.idx;
        layout[tempYName] = {
          domain: domain(tmp.utilRow - 1, nRows),
          title: {text: 'GPU \u00b0C', font: {color: '#ff6e40', size: 10}},
          overlaying: yax(tmp.utilRow), side: 'right', showgrid: false,
          autorange: true, minallowed: 30,
        };
      }
    }
    const cpuYName = cpuRow === 1 ? 'yaxis' : 'yaxis' + cpuRow;
    layout[cpuYName].title = {text: '<span style="color:#ffcc00">CPU %</span>', font: {size: 10}};
    layout[cpuYName].range = [0, 105];
    const diskYName = diskRow === 1 ? 'yaxis' : 'yaxis' + diskRow;
    layout[diskYName].title = {text: '<span style="color:#ff9800">Disk I/O MB/s</span>', font: {size: 10}};
    layout[diskYName].rangemode = 'tozero';
    // Network secondary y-axis (log) overlaying disk row
    const netYName = 'yaxis' + netYIdx;
    layout[netYName] = {
      domain: domain(diskRow - 1, nRows),
      title: {text: 'Net MB/s (log)', font: {color: '#18ffff', size: 10}},
      overlaying: yax(diskRow), side: 'right', showgrid: false,
      type: 'log', autorange: true,
    };

    // Create per-group legends aligned to their chart rows (grow downward)
    const legendStyle = {orientation: 'v', xanchor: 'left', x: 1.01, font: {size: 10}, bgcolor: 'rgba(13,13,20,0.8)', tracegroupgap: 1, itemsizing: 'constant', valign: 'top'};
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        const memDom = domain(2 * gi, nRows);
        const legName = gi === 0 ? 'legend' : 'legend' + (gi + 1);
        layout[legName] = Object.assign({}, legendStyle, {yanchor: 'top', y: memDom[1]});
      }
    }
    const cpuDom = domain(nRows - 2, nRows);
    layout[sysLegName] = Object.assign({}, legendStyle, {yanchor: 'top', y: cpuDom[1]});

    // Add background shapes and labels
    const axPairs = [];
    for (let r = 1; r <= nRows; r++) {
      axPairs.push([xax(r), yax(r)]);
    }
    for (let i = 0; i < axPairs.length; i++) {
      layout.shapes.push({
        type: 'rect', x0: 0, x1: 1, y0: 0, y1: 1,
        xref: axPairs[i][0] + ' domain', yref: axPairs[i][1] + ' domain',
        fillcolor: bgColors[i % 2], line: {width: 0}, layer: 'below',
      });
      rowAnnotations.push({index: layout.annotations.length, baseText: rowLabels[i], group: rowGroups[i]});
      layout.annotations.push({
        text: '  ' + rowLabels[i] + '  ', x: 0.0, y: 0.97,
        xref: axPairs[i][0] + ' domain', yref: axPairs[i][1] + ' domain',
        showarrow: false, font: {size: 11, color: '#c0c0d0', family: 'monospace'},
        xanchor: 'left', yanchor: 'top',
        bgcolor: 'rgba(30,30,50,0.92)', bordercolor: '#444466', borderwidth: 1, borderpad: 3,
      });
    }

    // GPU total memory ceiling line per GPU
    if (hasGpu) {
      for (let gi = 0; gi < gpuCount; gi++) {
        const maxGib = msg.gpu_mem_total_gib[gi] || 1;
        const yr = yax(2 * gi + 1);
        const xr = xax(2 * gi + 1);
        layout.shapes.push({
          type: 'line', x0: 0, x1: 1, y0: maxGib, y1: maxGib,
          xref: xr + ' domain', yref: yr,
          line: {color: '#ff5252', width: 2},
        });
        layout.annotations.push({
          text: maxGib.toFixed(1) + ' GiB', x: 1, y: maxGib,
          xref: xr + ' domain', yref: yr, showarrow: false,
          font: {size: 10, color: '#ff5252'}, xanchor: 'right',
        });
      }
    }

    Plotly.newPlot(gd, traces, layout, {
      displayModeBar: true, scrollZoom: true, responsive: true,
      toImageButtonOptions: {format: 'png', filename: 'gpu_monitor', height: 1080, width: 1920, scale: 2},
    });

    // Sync zoom/pan across all x-axes and freeze auto-scroll on manual interaction
    gd.on('plotly_relayout', function(ev) {
      if (syncing) return;
      // Detect user-initiated x-range change (zoom/pan)
      let newRange = null;
      for (const key in ev) {
        const m = key.match(/^xaxis(\d*)\.range\[(\d)\]$/);
        if (m) {
          if (!newRange) newRange = [null, null];
          newRange[parseInt(m[2])] = ev[key];
        }
      }
      if (!newRange || newRange[0] === null || newRange[1] === null) return;
      // Freeze auto-scroll
      paused = true;
      viewMinutes = 0;
      document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
      document.getElementById('pause-btn').textContent = 'Resume';
      document.getElementById('pause-btn').style.backgroundColor = '#ff5252';
      document.getElementById('status').textContent = 'Zoomed (click a time button or Resume)';
      // Sync all x-axes to the same range
      syncing = true;
      const gc = config.gpu_count || 0;
      const nr = (gc > 0 ? gc * 2 : 0) + 2;
      const upd = {};
      for (let r = 1; r <= nr; r++) {
        const ax = r === 1 ? 'xaxis' : 'xaxis' + r;
        upd[ax + '.range'] = [newRange[0], newRange[1]];
      }
      Plotly.relayout(gd, upd).then(function() { syncing = false; });
    });

    gd.on('plotly_doubleclick', function() {
      viewMinutes = 2;
      paused = false;
      document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
      document.querySelector('.range-btn[data-min="2"]').classList.add('active');
      document.getElementById('pause-btn').textContent = 'Pause';
      document.getElementById('pause-btn').style.backgroundColor = '#ffcc00';
      socket.emit('view_minutes', viewMinutes);
      document.getElementById('status').textContent = 'Live';
      doScroll();
      return false;
    });
  }

  function doScroll() {
    if (paused || !gd || viewMinutes <= 0) return;
    const now = new Date();
    const start = new Date(now.getTime() - viewMinutes * 60000);
    const upd = {};
    const gc = config.gpu_count || 0;
    const nr = (gc > 0 ? gc * 2 : 0) + 2;
    for (let r = 1; r <= nr; r++) {
      const ax = r === 1 ? 'xaxis' : 'xaxis' + r;
      upd[ax + '.range'] = [start, now];
    }
    Plotly.relayout(gd, upd);
  }

  setInterval(function() {
    const now = Date.now();
    const elapsed = (now - lastRateCheck) / 1000;
    if (elapsed > 0) {
      rate_main = ((cur_srv_main - prev_srv_main) / elapsed).toFixed(1);
      rate_disk = ((cur_srv_disk - prev_srv_disk) / elapsed).toFixed(1);
      rate_pcie_per_gpu = [];
      let pcieSum = 0;
      for (let i = 0; i < cur_srv_pcie.length; i++) {
        const prev = (prev_srv_pcie[i] || 0);
        const r = ((cur_srv_pcie[i] - prev) / elapsed);
        rate_pcie_per_gpu.push(r.toFixed(1));
        pcieSum += r;
      }
      rate_pcie = cur_srv_pcie.length > 0
        ? (pcieSum / cur_srv_pcie.length).toFixed(1) : '0.0';
    }
    prev_srv_main = cur_srv_main;
    prev_srv_pcie = cur_srv_pcie.slice();
    prev_srv_disk = cur_srv_disk;
    lastRateCheck = now;
    const el = document.getElementById('status');
    const base = paused ? 'Paused' : 'Live';
    const pcieRates = rate_pcie_per_gpu.map((r, i) => 'gpu' + i + ' ' + r).join(', ');
    el.textContent = base + '  |  main ' + rate_main + '/s  pcie [' + pcieRates + ']/s  disk ' + rate_disk + '/s';
    if (gd && rowAnnotations.length > 0) {
      let changed = false;
      for (const ra of rowAnnotations) {
        let rateStr = '';
        if (ra.group === 'main_pcie') rateStr = rate_main + ' samples/s, ' + rate_pcie + ' samples/s';
        else if (ra.group === 'main') rateStr = rate_main + ' samples/s';
        else if (ra.group === 'disk') rateStr = rate_disk + ' samples/s';
        else if (ra.group === 'pcie') rateStr = rate_pcie + ' samples/s';
        const newText = '  ' + ra.baseText + '<br>' + rateStr + '  ';
        if (gd.layout.annotations[ra.index].text !== newText) {
          gd.layout.annotations[ra.index].text = newText;
          changed = true;
        }
      }
      if (changed) {
        syncing = true;
        Plotly.relayout(gd, {});
        syncing = false;
      }
    }
  }, 5000);

  function startScrollLoop() {
    // Scrolling now happens in delta handler after extendTraces
  }

  // --- Controls ---
  document.querySelectorAll('.range-btn').forEach(btn => {
    btn.addEventListener('click', function() {
      document.querySelectorAll('.range-btn').forEach(b => b.classList.remove('active'));
      this.classList.add('active');
      viewMinutes = parseInt(this.dataset.min);
      paused = false;
      document.getElementById('pause-btn').textContent = 'Pause';
      document.getElementById('pause-btn').style.backgroundColor = '#ffcc00';
      document.getElementById('status').textContent = 'Live';
      socket.emit('view_minutes', viewMinutes);
      doScroll();
      if (viewMinutes === 0 && gd) {
        // Reset to autorange
        const gc2 = config.gpu_count || 0;
        const nr2 = (gc2 > 0 ? gc2 * 2 : 0) + 2;
        const upd = {};
        for (let r = 1; r <= nr2; r++) {
          const ax = r === 1 ? 'xaxis' : 'xaxis' + r;
          upd[ax + '.autorange'] = true;
        }
        Plotly.relayout(gd, upd);
      }
    });
  });

  document.getElementById('pause-btn').addEventListener('click', function() {
    paused = !paused;
    this.textContent = paused ? 'Resume' : 'Pause';
    this.style.backgroundColor = paused ? '#ff5252' : '#ffcc00';
    if (!paused) {
      // Resuming: restore viewMinutes from active button, default 2m
      const activeBtn = document.querySelector('.range-btn.active');
      viewMinutes = activeBtn ? parseInt(activeBtn.dataset.min) : 2;
      if (!activeBtn) {
        document.querySelector('.range-btn[data-min="2"]').classList.add('active');
      }
      document.getElementById('status').textContent = 'Live';
      socket.emit('view_minutes', viewMinutes);
    }
  });

  document.getElementById('snapshot-btn').addEventListener('click', function() {
    if (!gd) return;
    const d = new Date();
    const pad = n => String(n).padStart(2, '0');
    const fname = d.getFullYear() + '-' + pad(d.getMonth()+1) + '-' + pad(d.getDate()) +
      '_' + pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds()) + '_GPU';
    Plotly.downloadImage(gd, {format: 'png', width: 1920, height: 1080, scale: 2, filename: fname});
  });

  function hexToRgba(hex, alpha) {
    const h = hex.replace('#', '');
    return 'rgba(' + parseInt(h.substring(0,2),16) + ',' +
      parseInt(h.substring(2,4),16) + ',' + parseInt(h.substring(4,6),16) + ',' + alpha + ')';
  }
  function hexToHue(hex) {
    const h = hex.replace('#', '');
    const r = parseInt(h.substring(0,2),16)/255;
    const g = parseInt(h.substring(2,4),16)/255;
    const b = parseInt(h.substring(4,6),16)/255;
    const mx = Math.max(r,g,b), mn = Math.min(r,g,b), d = mx - mn;
    if (d === 0) return {hue: 0, sat: 0};
    let hue;
    if (mx === r) hue = ((g - b) / d + 6) % 6;
    else if (mx === g) hue = (b - r) / d + 2;
    else hue = (r - g) / d + 4;
    return {hue: hue * 60, sat: d / mx};
  }
  function findSimilarColors(colors, threshold) {
    const needs = new Set();
    const hsv = colors.map(c => hexToHue(c));
    for (let i = 0; i < hsv.length; i++) {
      if (hsv[i].sat < 0.15) continue;
      for (let j = i + 1; j < hsv.length; j++) {
        if (hsv[j].sat < 0.15) continue;
        let dh = Math.abs(hsv[i].hue - hsv[j].hue);
        if (dh > 180) dh = 360 - dh;
        if (dh < threshold) { needs.add(i); needs.add(j); }
      }
    }
    return needs;
  }
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Flask + SocketIO Server
# ---------------------------------------------------------------------------


def build_server(collector: MetricsCollector, args):
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "gpu_monitor"
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode="threading",
        ping_timeout=60,
        ping_interval=25,
        max_http_buffer_size=50 * 1024 * 1024,
    )

    ui_state = {"paused": False}

    @app.route("/")
    def index():
        return Response(HTML_PAGE, mimetype="text/html")

    @socketio.on("connect")
    def handle_connect():
        sid = flask_request.sid

        def send_init():
            for _ in range(50):
                with collector.lock:
                    if len(collector.ts_main) > 5:
                        break
                time.sleep(0.1)
            data = collector.snapshot_full()
            socketio.emit("init", data, to=sid)

        threading.Thread(target=send_init, daemon=True).start()

    @socketio.on("request_init")
    def handle_request_init():
        data = collector.snapshot_full()
        socketio.emit("init", data, to=flask_request.sid)

    @socketio.on("pause")
    def handle_pause(is_paused):
        ui_state["paused"] = bool(is_paused)

    @socketio.on("view_minutes")
    def handle_view_minutes(vm):
        ui_state["view_minutes"] = int(vm)

    # Push interval and thinning step per zoom level (view_minutes).
    # Wider views push less often and thin more aggressively.
    _PUSH_POLICY = {
        1: (0.075, 1),
        2: (0.100, 1),
        5: (0.200, 2),
        10: (0.500, 4),
        15: (1.000, 8),
    }

    def push_loop():
        """Background thread that pushes deltas to all clients."""
        last_main = 0
        last_pcie: list[int] = [0] * collector.gpu_count
        last_disk = 0
        while True:
            vm = ui_state.get("view_minutes", 2)
            push_sec, step = _PUSH_POLICY.get(vm, (0.100, 1))
            time.sleep(push_sec)
            with collector.lock:
                cm = collector.counter_main
                cp = list(collector.counter_pcie)
                cd = collector.counter_disk
            if cm <= last_main and cp == last_pcie and cd <= last_disk:
                continue
            try:
                delta = collector.snapshot_delta(
                    last_main, last_pcie, last_disk, step=step
                )
                last_main = cm
                last_pcie = cp
                last_disk = cd
                socketio.emit("delta", delta)
            except Exception as e:
                print(f"[push_loop] error: {e}", flush=True)
                raise

    return app, socketio, ui_state, push_loop


def main():
    args = parse_args()
    gpu_count = 0
    if HAS_NVML:
        gpu_count = pynvml.nvmlDeviceGetCount()

    main_sec = args.main_interval / 1000.0
    disk_sec = args.disk_interval / 1000.0
    print(
        f"Main collect : {args.main_interval}ms (CPU + CPU procs + GPU mem/util/temp + network)"
    )
    print(
        f"PCIe collect : 10/s ({gpu_count} subprocess{'es' if gpu_count != 1 else ''})"
    )
    print(f"Disk collect : {args.disk_interval}ms (aggregate disk I/O)")
    print("Push interval: dynamic (75ms@1m .. 1000ms@15m)")
    print(
        f"Rolling window  : {args.window}s ({args.window // 60}m {args.window % 60}s)"
    )

    if HAS_NVML:
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU {i}: {name} ({mem.total / (1024**3):.1f} GiB)")
    else:
        print("No NVIDIA GPUs detected -- CPU + Disk only")
    print()

    collector = MetricsCollector(
        window_sec=args.window,
        main_interval_ms=args.main_interval,
        disk_interval_ms=args.disk_interval,
        top_n=args.top_n,
    )
    collector.load_state()

    app, socketio, ui_state, push_loop = build_server(collector, args)

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
        # 1 main subprocess: CPU + GPU mem/util/temp + network
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
        while True:
            ready = multiprocessing.connection.wait(pipes, timeout=0.5)
            for conn in ready:
                msg = conn.recv()
                tag = msg[0]
                if tag == "pcie":
                    collector.ingest_pcie(msg[1], msg[2], msg[3], msg[4])
                elif tag == "main":
                    collector.ingest_main(
                        msg[1], msg[2], msg[3], msg[4], msg[5], msg[6]
                    )
                elif tag == "disk":
                    collector.ingest_disk(msg[1], msg[2], msg[3])

    for w in workers:
        w.start()
    threading.Thread(target=_poll_pipes, daemon=True).start()
    threading.Thread(target=push_loop, daemon=True).start()

    def _shutdown(signum, _frame):
        name = signal.Signals(signum).name
        print(f"\n[{name}] saving metrics before exit...", flush=True)
        collector.save_state()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print(f"Open http://{args.host}:{args.port} in your browser")
    socketserver.TCPServer.allow_reuse_address = True
    socketio.run(app, host=args.host, port=args.port, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
