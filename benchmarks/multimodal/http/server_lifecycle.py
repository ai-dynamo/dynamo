# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Spawn the local media server as a subprocess; tear it down on exit.

Used by ``sweep.py`` to bring up a fresh server with a different
``--processing-time-mean-ms`` per iteration without leaking processes.
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _wait_ready(port: int, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Exception | None = None
    url = f"http://localhost:{port}/test"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last_err = e
        time.sleep(0.2)
    raise TimeoutError(
        f"local_media_server not ready in {timeout_s}s on :{port}: {last_err!r}"
    )


@contextmanager
def local_media_server(
    *,
    port: int,
    image_url: str,
    mean_ms: float,
    variance_ms: float = 0.0,
    ready_timeout_s: float = 15.0,
    log_path: Path | None = None,
) -> Iterator[str]:
    cmd = [
        sys.executable,
        "-m",
        "benchmarks.multimodal.local_media_server.main",
        "--image",
        f"test:{image_url}",
        "--port",
        str(port),
        "--processing-time-mean-ms",
        str(mean_ms),
        "--processing-time-variance-ms",
        str(variance_ms),
    ]
    log_fd: object
    if log_path is not None:
        log_fd = open(log_path, "ab")
    else:
        log_fd = subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdout=log_fd, stderr=subprocess.STDOUT)
    try:
        _wait_ready(port, ready_timeout_s)
        yield f"http://localhost:{port}/test"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)
        if log_fd is not subprocess.DEVNULL:
            log_fd.close()  # type: ignore[union-attr]
