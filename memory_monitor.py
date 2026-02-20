"""
Shared memory monitoring utilities for Dynamo repro components.

Enable with:
  DYNAMO_MEMORY_PROFILE=1  (default: 1)
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Optional, Tuple


def _read_proc_status_bytes() -> Tuple[int, int]:
    """
    Return (rss_bytes, vms_bytes) for the current process.

    Linux-only (this repro harness targets Linux/k8s). Falls back to (0, 0) if unreadable.
    """
    try:
        rss = 0
        vms = 0
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                # Lines look like: "VmRSS:\t  12345 kB"
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) * 1024
                elif line.startswith("VmSize:"):
                    vms = int(line.split()[1]) * 1024
        return rss, vms
    except Exception:  # pragma: no cover
        return 0, 0


class MemoryMonitor:
    """Memory monitor that can be shared across components."""

    def __init__(self, component_name: str):
        self.request_count = 0
        self.start_time = time.time()
        self.component_name = component_name
        self.freed_last_mb: float = 0.0

    @property
    def initial_memory(self) -> int:
        if not hasattr(self, "_initial_memory"):
            self._initial_memory = _read_proc_status_bytes()[0]
        return self._initial_memory

    def log_memory(self, label: str = "") -> None:
        rss_bytes, vms_bytes = _read_proc_status_bytes()
        rss_mb = rss_bytes / 1024 / 1024
        vms_mb = vms_bytes / 1024 / 1024
        rss_growth = (rss_bytes - self.initial_memory) / 1024 / 1024
        uptime = time.time() - self.start_time

        print(
            f"[{self.component_name}] [{uptime:.1f}s] {label} "
            f"Memory: RSS={rss_mb:.2f}MB (+{rss_growth:.2f}MB), "
            f"VMS={vms_mb:.2f}MB, Requests={self.request_count}, "
            f"Last GC freed={self.freed_last_mb:.2f}MB"
        )

    def increment_request(self) -> None:
        self.request_count += 1
        _ = self.initial_memory

        # Log every 5000 requests.
        if self.request_count % 5000 == 0:
            threading.Thread(
                target=self.log_memory,
                args=(f"After {self.request_count} requests:",),
                daemon=True,
            ).start()


def create_monitor(component_name: str) -> Optional[MemoryMonitor]:
    if os.getenv("DYNAMO_MEMORY_PROFILE", "1") == "1":
        return MemoryMonitor(component_name)
    return None


def setup_background_monitor(
    monitor: Optional[MemoryMonitor],
) -> Optional[asyncio.Task]:
    if not monitor:
        print("Memory monitoring not enabled.")
        return None

    async def background_monitor() -> None:
        while True:
            await asyncio.sleep(15)
            monitor.log_memory("Background check:")

    return asyncio.create_task(background_monitor())
