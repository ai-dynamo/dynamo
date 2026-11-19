# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the sweep harness.

The harness must not attribute a result to a backend. It used to select a
backend by writing ``DYN_HTTP_BACKEND``, and it labeled each result with the
name that the caller passed. The facade now logs a warning for any value other
than ``aiohttp`` and uses aiohttp. A run requested as ``httpx`` therefore used
``AiohttpClient`` but printed under an ``httpx`` column, so both columns of the
table measured aiohttp. There is one backend now, so the harness takes no
selector and prints no backend label.

The harness must also reach its own media server, and it must not exit 0 when
it measured nothing.
"""

from __future__ import annotations

import contextlib
import http.server
import os
import socket
import threading

import pytest

from benchmarks.multimodal.http import sweep
from benchmarks.multimodal.http.runner import RunResult, run_one
from dynamo.common.http import _ssrf_resolver

# Leave these tests unmarked. The root conftest.py then adds ``pre_merge``,
# ``gpu_0`` and ``defaulted``, and the dynamo-runtime pipeline runs tests with
# ``defaulted`` in its CPU parallel job.


@pytest.mark.asyncio
async def test_sweep_runs_each_cell_once_and_prints_one_column(
    monkeypatch, capsys
) -> None:
    """The sweep used to run each cell twice and print an ``httpx`` column.

    Only the media server and the URL list are stubbed. The real run, summary
    and report code runs with no URLs, so no request leaves the process.
    """
    runs = []
    real_run_one = sweep.run_one

    async def counting_run_one(*args, **kwargs):
        runs.append(args)
        return await real_run_one(*args, **kwargs)

    @contextlib.contextmanager
    def fake_media_server(**kwargs):
        yield "http://media.invalid/test"

    monkeypatch.setattr(sweep, "run_one", counting_run_one)
    monkeypatch.setattr(sweep, "local_media_server", fake_media_server)
    monkeypatch.setattr(sweep, "gen_urls", lambda seeds, n: [])
    monkeypatch.delenv("DYN_HTTP_BACKEND", raising=False)

    args = sweep.parse_args(
        [
            "--server-processing-time-means-ms",
            "10,20",
            "--request-rate",
            "5,7",
            "--requests",
            "3",
        ]
    )
    assert await sweep._run_sweep(args) == 0

    # Two request rates times two delays.
    assert len(runs) == 4
    out = capsys.readouterr().out
    for label in ("httpx", "aiohttp", "backend"):
        assert label not in out
    grid_headers = [
        line for line in out.splitlines() if line.lstrip().startswith("mean_ms |")
    ]
    assert len(grid_headers) == 2
    assert all(header.count(" | ") == 1 for header in grid_headers)


@pytest.mark.asyncio
async def test_run_one_leaves_dyn_http_backend_unset(monkeypatch) -> None:
    """This catches a write to the environment, which the sweep test does not see."""
    monkeypatch.delenv("DYN_HTTP_BACKEND", raising=False)
    result = await run_one([], timeout=1.0, request_rate=100.0)
    assert result.n == 0
    assert "DYN_HTTP_BACKEND" not in os.environ


@pytest.mark.asyncio
async def test_run_one_does_not_clobber_an_operator_set_backend(monkeypatch) -> None:
    """This catches a run that deletes or changes the value that an operator set."""
    monkeypatch.setenv("DYN_HTTP_BACKEND", "aiohttp")
    await run_one([], timeout=1.0, request_rate=100.0)
    assert os.environ["DYN_HTTP_BACKEND"] == "aiohttp"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("labels", "expected_exit"),
    [
        (["HttpConnectionError"] * 3, 1),
        (["HttpConnectionError", "HttpConnectionError", "success"], 0),
    ],
    ids=["every-request-failed", "one-succeeded"],
)
async def test_sweep_exit_status_flags_a_pair_that_measured_nothing(
    monkeypatch, capsys, labels, expected_exit
) -> None:
    """A pair in which every request failed prints zero latencies, so the sweep
    must not exit 0. One success is a measurement, and the exit stays 0."""

    async def canned_run_one(urls, timeout, request_rate):
        samples = [(0.01, label) for label in labels]
        return RunResult(n=len(samples), wall_s=0.1, samples=samples)

    @contextlib.contextmanager
    def fake_media_server(**kwargs):
        yield "http://media.invalid/test"

    monkeypatch.setattr(sweep, "run_one", canned_run_one)
    monkeypatch.setattr(sweep, "local_media_server", fake_media_server)
    args = sweep.parse_args(
        [
            "--server-processing-time-means-ms",
            "10",
            "--request-rate",
            "5",
            "--requests",
            "3",
        ]
    )

    assert await sweep._run_sweep(args) == expected_exit
    err = capsys.readouterr().err
    failed = "every request failed at request_rate=5 mean_ms=10: HttpConnectionError=3"
    assert (failed in err) is (expected_exit == 1)


class _LoopbackResolver:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def resolve(self, host, port=0, family=socket.AF_INET):
        return [
            {
                "hostname": host,
                "host": "127.0.0.1",
                "port": port,
                "family": socket.AF_INET,
                "proto": 0,
                "flags": 0,
            }
        ]

    async def close(self) -> None:
        pass


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_main_reaches_the_local_media_server(monkeypatch) -> None:
    """The media server runs on localhost, which the fetch path refuses by
    default. The sweep used to fail every request there and still exit 0.

    This server answers on loopback under a host name, so each fetch passes
    the client's connect-time address check, as the real sweep's localhost URL
    does. An IP literal would skip that check, and the test would pass without
    the fix.
    """
    hits: list[str] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            hits.append(self.path)
            body = b"image"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    @contextlib.contextmanager
    def fake_media_server(**kwargs):
        yield f"http://media.sweep.test:{server.server_port}/test"

    monkeypatch.setattr(sweep, "local_media_server", fake_media_server)
    monkeypatch.setattr(_ssrf_resolver, "DefaultResolver", _LoopbackResolver)
    for name in (
        "http_proxy",
        "https_proxy",
        "all_proxy",
        "no_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)
    # Set, then delete: monkeypatch then restores the variable as it was before
    # the test, which also removes the value that main() writes.
    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "")
    monkeypatch.delenv("DYN_MM_ALLOW_INTERNAL")
    # Start the server last, so that no setup step can fail and leave it
    # running.
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        exit_status = sweep.main(
            [
                "--server-processing-time-means-ms",
                "1",
                "--request-rate",
                "50",
                "--requests",
                "3",
                "--timeout",
                "5",
            ]
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert len(hits) == 3
    assert exit_status == 0
