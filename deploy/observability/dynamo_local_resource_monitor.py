#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Local Resource Monitor — per-process resource metrics exporter for Prometheus.

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
  - Main process: polls pipes, updates Prometheus gauges

Exposes a /metrics endpoint in Prometheus exposition format.

Usage:
    python3 dynamo_local_resource_monitor.py [--port 8051] [--host 0.0.0.0]
"""

import argparse
import importlib.util
import multiprocessing
import os
import signal
import sys
import threading
import time

_REQUIRED_PACKAGES = {
    "psutil": "psutil",
    "pynvml": "nvidia-ml-py",
    "prometheus_client": "prometheus-client",
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
from prometheus_client import Gauge, start_http_server  # noqa: E402

HAS_NVML = False
try:
    pynvml.nvmlInit()
    HAS_NVML = True
except pynvml.NVMLError as e:
    print(f"NVML init failed ({e}) -- GPU monitoring disabled")


def parse_args():
    p = argparse.ArgumentParser(
        description="GPU/CPU/Disk/Network Prometheus metrics exporter"
    )
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

    updater = PrometheusUpdater(gpu_count)

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
                elif tag == "main":
                    updater.ingest_main(msg[2], msg[3], msg[4], msg[5], msg[6])
                elif tag == "disk":
                    updater.ingest_disk(msg[2], msg[3])

    for w in workers:
        w.start()
    threading.Thread(target=_poll_pipes, daemon=True).start()

    # Start Prometheus HTTP server
    start_http_server(args.port, addr=args.host)
    print(f"Prometheus metrics at http://{args.host}:{args.port}/metrics")

    def _shutdown(signum, _frame):
        name = signal.Signals(signum).name
        print(f"\n[{name}] shutting down.", flush=True)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Block main thread (workers and HTTP server are in background threads)
    try:
        while True:
            time.sleep(3600)
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
