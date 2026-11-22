# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A long GIL hold must not stall the HTTP server of a Python-engine service.

Every streamed chunk crosses the Rust/Python bridge. When the bridge takes the GIL on a
Tokio runtime worker, a thread that holds the GIL for seconds parks every worker that is
streaming; with more streams in flight than runtime worker threads, routes that never
touch Python (`/v1/models`, `/health`, `/live`) stop answering too. Here: the runtime
limited to WORKER_THREADS workers, STREAMS streams (more than every Tokio worker in the
process), the GIL held for HOLD_S seconds by another thread, and a probe in a separate
process (its own GIL) that must keep getting fast answers from `/v1/models`.
"""

import asyncio
import contextlib
import json
import subprocess
import sys
import textwrap
import threading
import time
from typing import AsyncGenerator, Dict

import pytest

from dynamo.llm import HttpAsyncEngine, HttpService
from dynamo.runtime import DistributedRuntime

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]

MODEL = "gil_test_model"
WORKER_THREADS = 2
STREAMS = 32
HOLD_S = 2.5
PROBE_TIMEOUT_S = 1.0

# Runs in a child process so that its HTTP client has a GIL of its own.
PROBE_SCRIPT = textwrap.dedent(
    """
    import http.client, json, sys, threading, time
    port, model, streams, probe_timeout, duration = (
        int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]))
    body = json.dumps({"model": model, "stream": True,
                       "messages": [{"role": "user", "content": "hi"}]})
    started = threading.Semaphore(0)

    def stream():
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
        c.request("POST", "/v1/chat/completions", body, {"content-type": "application/json"})
        r = c.getresponse()
        r.read1(1)
        started.release()
        while r.read1(65536):
            pass

    for _ in range(streams):
        threading.Thread(target=stream, daemon=True).start()
    for _ in range(streams):
        started.acquire()
    print("READY", flush=True)

    probes = []
    end = time.monotonic() + duration
    while time.monotonic() < end:
        started_at, t0 = time.time(), time.monotonic()
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=probe_timeout)
            c.request("GET", "/v1/models")
            ok = c.getresponse().status == 200
        except Exception:
            ok = False
        probes.append([started_at, ok, time.monotonic() - t0])
        time.sleep(0.05)
    print(json.dumps(probes), flush=True)
    """
)


class SlowStreamEngine:
    """Streams many small chunks, so a bridge crossing is always in flight."""

    async def generate(self, request: Dict, context) -> AsyncGenerator[Dict, None]:
        for i in range(4000):
            if context.is_stopped():
                return
            yield {
                "_dynamo_annotated": True,
                "data": {
                    "id": f"chatcmpl-{context.id()}",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": MODEL,
                    "choices": [
                        {"index": 0, "delta": {"content": "x"}, "finish_reason": None}
                    ],
                },
            }
            await asyncio.sleep(0.002)


def _hold_gil(seconds: float) -> None:
    """Hold the GIL without releasing it, like a long call into a C extension.

    Runs on its own thread: the event loop keeps running the engine's generators, so the
    bridge keeps crossing into Python, as it does when a chat processor's worker thread or
    a C extension holds the GIL in production."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(seconds + 5)
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            pass
    finally:
        sys.setswitchinterval(previous)


@pytest.mark.asyncio
@pytest.mark.forked
@pytest.mark.timeout(90)
async def test_gil_hold_does_not_stall_http_routes(
    unused_tcp_port, tmp_path, monkeypatch
):
    monkeypatch.setenv("DYN_FILE_KV", str(tmp_path))
    monkeypatch.setenv("DYN_RUNTIME_NUM_WORKER_THREADS", str(WORKER_THREADS))
    loop = asyncio.get_running_loop()
    runtime = DistributedRuntime(loop, "file", "tcp", event_plane="zmq")
    service = HttpService(port=unused_tcp_port)
    engine = HttpAsyncEngine(SlowStreamEngine().generate, loop)
    service.add_chat_completions_model(MODEL, "abc123", engine)
    service.enable_endpoint("chat", True)
    server = asyncio.ensure_future(service.run(runtime))
    probe = None
    try:
        for _ in range(100):
            with contextlib.suppress(OSError):
                _, writer = await asyncio.open_connection("127.0.0.1", unused_tcp_port)
                writer.close()
                break
            await asyncio.sleep(0.1)

        probe = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            PROBE_SCRIPT,
            str(unused_tcp_port),
            MODEL,
            str(STREAMS),
            str(PROBE_TIMEOUT_S),
            str(HOLD_S + 3.0),
            stdout=subprocess.PIPE,
        )
        assert probe.stdout is not None
        ready = await asyncio.wait_for(probe.stdout.readline(), timeout=30)
        assert ready.strip() == b"READY"
        await asyncio.sleep(0.5)

        holder = threading.Thread(target=_hold_gil, args=(HOLD_S,), daemon=True)
        hold_start = time.time()
        holder.start()
        while holder.is_alive():
            await asyncio.sleep(0.05)
        hold_end = time.time()

        out = await asyncio.wait_for(probe.stdout.readline(), timeout=30)
        probes = json.loads(out)
        during = [p for p in probes if hold_start <= p[0] <= hold_end]
        assert during, "no probe ran while the GIL was held"
        failed = [p for p in during if not p[1]]
        slowest = max(p[2] for p in during)
        assert not failed and slowest < PROBE_TIMEOUT_S, (
            f"/v1/models stalled while the GIL was held: {len(failed)}/{len(during)} "
            f"probes failed, slowest {slowest:.2f}s"
        )
    finally:
        if probe is not None and probe.returncode is None:
            probe.kill()
        service.shutdown()
        server.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(server, timeout=10.0)
        runtime.shutdown()
