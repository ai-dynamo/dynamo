# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests: the sweep harness must not attribute a result to a backend.

The harness used to select a backend by writing ``DYN_HTTP_BACKEND``, and it
labeled each result with the name that the caller passed. The facade now logs
a warning for any value other than ``aiohttp`` and uses aiohttp. A run
requested as ``httpx`` therefore used ``AiohttpClient`` but printed under an
``httpx`` column, so both columns of the table measured aiohttp. There is one
backend now, so the harness takes no selector and prints no backend label.
"""

from __future__ import annotations

import contextlib
import os

import pytest

from benchmarks.multimodal.http import sweep
from benchmarks.multimodal.http.runner import run_one

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
