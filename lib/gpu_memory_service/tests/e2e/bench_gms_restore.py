#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark GMS restore with varying worker counts.

Saves GMS state once (if not already saved), then loops over worker counts
for restore-only, reporting per-phase timings.

Usage
-----
    # Save + benchmark (first run or if save dir is missing):
    python bench_gms_restore.py \\
        --save-dir /ramdisk/gms_bench_save_72b \\
        --model Qwen/Qwen2.5-72B-Instruct \\
        --workers 16 32 64

    # Restore-only benchmark (save dir already populated):
    python bench_gms_restore.py \\
        --save-dir /ramdisk/gms_bench_save_72b \\
        --restore-only \\
        --workers 16 32 64

Phase timing is extracted from log timestamps in the subprocess output:
    Disk+A (disk+alloc):  "Loading GMS state" → "Phase A complete"  (overlapped)
    Phase B (GPU copy):   "Phase A complete"  → "Phase B complete"
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")

# ---------------------------------------------------------------------------
# Process helpers (shared with e2e test)
# ---------------------------------------------------------------------------


class ManagedProcess:
    def __init__(self, cmd: list[str], log_path: Path, env: Optional[dict] = None):
        self._log_fh = log_path.open("w")
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            env=env or os.environ.copy(),
            preexec_fn=os.setsid,
        )
        logger.info("Started pid=%d: %s", self._proc.pid, " ".join(cmd))

    @property
    def pid(self) -> int:
        return self._proc.pid

    def is_running(self) -> bool:
        return self._proc.poll() is None

    def terminate(self, name: str = "") -> None:
        if not self.is_running():
            return
        try:
            os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            try:
                self._proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                self._proc.wait(timeout=5)
        except ProcessLookupError:
            pass
        self._log_fh.close()
        logger.info("Stopped %s", name or f"pid={self._proc.pid}")

    def tail(self, n: int = 20) -> str:
        try:
            lines = Path(self._log_fh.name).read_text().splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return "(unavailable)"


def _python() -> str:
    return sys.executable


def _gms_server_cmd(device: int = 0) -> list[str]:
    return [_python(), "-m", "gpu_memory_service", "--device", str(device)]


def _vllm_serve_cmd(model: str, port: int) -> list[str]:
    return [
        _python(),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--load-format",
        "gms",
        "--worker-cls",
        "gpu_memory_service.integrations.vllm.worker.GMSWorker",
        "--port",
        str(port),
        "--max-model-len",
        "512",
        "--disable-log-stats",
    ]


def _device_env(device: int) -> dict:
    """Return an env dict with CUDA_VISIBLE_DEVICES set to *device*."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(device)
    return env


def _gms_save_cmd(output_dir: Path, device: int = 0) -> list[str]:
    return [
        _python(),
        "-m",
        "gpu_memory_service.cli.storage_runner",
        "save",
        "--output-dir",
        str(output_dir),
        "--device",
        str(device),
        "--verbose",
    ]


def _wait_for_http(url: str, timeout: int) -> bool:
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(3)
    return False


def _wait_gpu_free(exclude_pids: set[int], timeout: int = 120) -> bool:
    if shutil.which("nvidia-smi") is None:
        logger.warning("nvidia-smi not found; skipping GPU idle wait")
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        if r.returncode != 0:
            logger.warning(
                "nvidia-smi exited with %d; skipping GPU idle wait", r.returncode
            )
            return True
        pids = {
            int(p.strip()) for p in r.stdout.strip().split("\n") if p.strip().isdigit()
        }
        if not (pids - exclude_pids):
            return True
        time.sleep(3)
    return False


def _gms_socket_path(device: int = 0) -> Path:
    from gpu_memory_service.common.utils import get_socket_path

    return Path(get_socket_path(device))


# ---------------------------------------------------------------------------
# Phase timing extraction from log output
# ---------------------------------------------------------------------------

_MARKERS = {
    "disk_start": re.compile(r"Loading GMS state:"),
    "phaseA_end": re.compile(r"Phase A complete"),
    "phaseB_end": re.compile(r"Phase B complete"),
}


def _ts(line: str) -> Optional[datetime]:
    m = _TIMESTAMP_RE.match(line)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S,%f")


def parse_phase_times(log_text: str) -> dict[str, float]:
    """Return {diskA_s, phaseB_s, total_s} from subprocess log text.

    Disk reads and Phase A (GMS VA allocation) run concurrently in the new
    pipeline, so only the combined duration (disk_start → Phase A complete)
    is measurable from logs.
    """
    times: dict[str, datetime] = {}
    for line in log_text.splitlines():
        for key, pat in _MARKERS.items():
            if key not in times and pat.search(line):
                ts = _ts(line)
                if ts:
                    times[key] = ts
    result: dict[str, float] = {}
    if "disk_start" in times and "phaseA_end" in times:
        result["diskA_s"] = (times["phaseA_end"] - times["disk_start"]).total_seconds()
    if "phaseA_end" in times and "phaseB_end" in times:
        result["phaseB_s"] = (times["phaseB_end"] - times["phaseA_end"]).total_seconds()
    if "disk_start" in times and "phaseB_end" in times:
        result["total_s"] = (times["phaseB_end"] - times["disk_start"]).total_seconds()
    return result


# ---------------------------------------------------------------------------
# Save phase
# ---------------------------------------------------------------------------


def run_save(
    save_dir: Path,
    model: str,
    device: int = 0,
    vllm_port: int = 18_700,
    startup_timeout: int = 1800,
    log_dir: Optional[Path] = None,
) -> None:
    """Load model via vLLM → GMS, save GMS state to disk, tear down."""
    if log_dir is None:
        log_dir = save_dir.parent / "gms_bench_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    gms: Optional[ManagedProcess] = None
    vllm: Optional[ManagedProcess] = None
    try:
        logger.info("=== Save phase: starting GMS + vLLM ===")
        gms = ManagedProcess(_gms_server_cmd(device), log_dir / "gms_save.log")
        time.sleep(2)
        assert gms.is_running(), f"GMS exited.\n{gms.tail()}"

        vllm = ManagedProcess(
            _vllm_serve_cmd(model, vllm_port),
            log_dir / "vllm_save.log",
            env=_device_env(device),
        )
        assert _wait_for_http(
            f"http://localhost:{vllm_port}/health", startup_timeout
        ), f"vLLM did not become healthy.\nvLLM:\n{vllm.tail()}\nGMS:\n{gms.tail()}"
        logger.info("vLLM healthy; saving GMS state → %s", save_dir)

        subprocess.run(_gms_save_cmd(save_dir, device), check=True)

        manifest = json.loads((save_dir / "manifest.json").read_text())
        n = len(manifest.get("allocations", []))
        total_gib = sum(a["aligned_size"] for a in manifest["allocations"]) / 1024**3
        logger.info("Saved %d allocations (%.2f GiB) → %s", n, total_gib, save_dir)

    finally:
        if vllm:
            vllm.terminate("vLLM")
        if gms:
            gms.terminate("GMS")
        _wait_gpu_free(exclude_pids={os.getpid()}, timeout=120)
        sock = _gms_socket_path(device)
        if sock.exists():
            sock.unlink()


# ---------------------------------------------------------------------------
# Restore benchmark
# ---------------------------------------------------------------------------


def _gms_load_cmd(input_dir: Path, device: int = 0, workers: int = 16) -> list[str]:
    return [
        _python(),
        "-m",
        "gpu_memory_service.cli.storage_runner",
        "load",
        "--input-dir",
        str(input_dir),
        "--device",
        str(device),
        "--workers",
        str(workers),
        "--verbose",
    ]


def _fadvise_dontneed(path: Path) -> None:
    """Tell the kernel to evict *path* from the page cache (POSIX_FADV_DONTNEED).

    Works inside an unprivileged container — no SYS_ADMIN capability required.
    Falls back silently if the libc call is unavailable.
    """
    _libc_name = ctypes.util.find_library("c")
    if not _libc_name:
        return
    _libc = ctypes.CDLL(_libc_name, use_errno=True)
    POSIX_FADV_DONTNEED = 4  # Linux constant
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            _libc.posix_fadvise(
                fd, ctypes.c_int64(0), ctypes.c_int64(0), POSIX_FADV_DONTNEED
            )
        finally:
            os.close(fd)
    except OSError:
        pass


def _evict_save_dir(save_dir: Path) -> None:
    """Evict all shard files and JSON metadata in *save_dir* from page cache."""
    evicted = 0
    for p in save_dir.rglob("*"):
        if p.is_file():
            _fadvise_dontneed(p)
            evicted += 1
    logger.info("fadvise(DONTNEED) called on %d files in %s", evicted, save_dir)


def bench_restore(
    save_dir: Path,
    workers: int,
    device: int = 0,
    log_dir: Optional[Path] = None,
    drop_cache: bool = False,
) -> dict:
    """Start fresh GMS, run restore with *workers*, return timing dict."""
    if log_dir is None:
        log_dir = save_dir.parent / "gms_bench_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((save_dir / "manifest.json").read_text())
    total_bytes = sum(a["aligned_size"] for a in manifest["allocations"])
    total_gib = total_bytes / 1024**3

    if drop_cache:
        _evict_save_dir(save_dir)

    gms: Optional[ManagedProcess] = None
    log_path = log_dir / f"gms_restore_w{workers}.log"
    restore_log_path = log_dir / f"restore_w{workers}.log"

    try:
        gms = ManagedProcess(_gms_server_cmd(device), log_path)
        time.sleep(2)
        assert gms.is_running(), f"GMS exited.\n{gms.tail()}"

        cmd = _gms_load_cmd(save_dir, device, workers)
        wall_t0 = time.monotonic()
        with open(restore_log_path, "w") as log_f:
            subprocess.run(
                cmd,
                check=True,
                capture_output=False,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
        wall_elapsed = time.monotonic() - wall_t0

        log_text = restore_log_path.read_text()
        times = parse_phase_times(log_text)
        times["wall_s"] = wall_elapsed
        times["total_gib"] = total_gib
        if "phaseB_s" in times and times["phaseB_s"] > 0:
            times["phaseB_gib_s"] = total_gib / times["phaseB_s"]
        if "total_s" in times and times["total_s"] > 0:
            times["total_gib_s"] = total_gib / times["total_s"]

        logger.info(
            "workers=%3d  diskA=%.2fs  phaseB=%.2fs  total=%.2fs  "
            "B_throughput=%.2f GiB/s",
            workers,
            times.get("diskA_s", float("nan")),
            times.get("phaseB_s", float("nan")),
            times.get("total_s", float("nan")),
            times.get("phaseB_gib_s", float("nan")),
        )
        return {"workers": workers, **times}

    finally:
        if gms:
            gms.terminate("GMS")
        _wait_gpu_free(exclude_pids={os.getpid()}, timeout=60)
        sock = _gms_socket_path(device)
        if sock.exists():
            sock.unlink()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GMS restore phases")
    parser.add_argument(
        "--save-dir",
        default="/ramdisk/gms_bench_save_72b",
        help="Persistent directory for saved GMS state",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-72B-Instruct",
        help="HuggingFace model id (used only if save needed)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[16, 32, 64],
        help="Worker counts to benchmark",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--vllm-port", type=int, default=18_700)
    parser.add_argument(
        "--restore-only",
        action="store_true",
        help="Skip save phase even if save-dir is missing",
    )
    parser.add_argument(
        "--drop-cache",
        action="store_true",
        help="Call fadvise(DONTNEED) on shard files before each run "
        "to evict them from the page cache (cold-read measurement)",
    )
    args = parser.parse_args()

    save_dir = Path(args.save_dir)

    # Save phase (if needed)
    if not (save_dir / "manifest.json").exists():
        if args.restore_only:
            sys.exit(f"Save dir {save_dir} missing but --restore-only set.")
        logger.info("No save data found; running save phase first.")
        run_save(save_dir, args.model, args.device, args.vllm_port)
    else:
        manifest = json.loads((save_dir / "manifest.json").read_text())
        total_gib = sum(a["aligned_size"] for a in manifest["allocations"]) / 1024**3
        logger.info(
            "Using existing save at %s (%.2f GiB, %d allocs)",
            save_dir,
            total_gib,
            len(manifest["allocations"]),
        )

    # Restore benchmark
    logger.info(
        "=== Starting restore benchmark: workers=%s drop_cache=%s ===",
        args.workers,
        args.drop_cache,
    )
    results = []
    for w in args.workers:
        logger.info("--- workers=%d ---", w)
        r = bench_restore(save_dir, w, args.device, drop_cache=args.drop_cache)
        results.append(r)

    # Summary table
    print()
    print(
        f"{'workers':>8}  {'diskA_s':>8}  {'phaseB_s':>9}  "
        f"{'total_s':>8}  {'B_GiB/s':>8}  {'tot_GiB/s':>10}"
    )
    print("-" * 68)
    for r in results:
        print(
            f"{r['workers']:>8}  "
            f"{r.get('diskA_s', float('nan')):>8.2f}  "
            f"{r.get('phaseB_s', float('nan')):>9.2f}  "
            f"{r.get('total_s', float('nan')):>8.2f}  "
            f"{r.get('phaseB_gib_s', float('nan')):>8.2f}  "
            f"{r.get('total_gib_s', float('nan')):>10.2f}"
        )
    print()


if __name__ == "__main__":
    main()
