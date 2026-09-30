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

import dataclasses
import inspect
import os

import pytest

from benchmarks.multimodal.http.runner import RunResult, run_one
from benchmarks.multimodal.http.stats import Summary

# Leave these tests unmarked. The root conftest.py then adds ``pre_merge``,
# ``gpu_0`` and ``defaulted``, and the dynamo-runtime pipeline runs tests with
# ``defaulted`` in its CPU parallel job.


def test_run_one_takes_no_backend_selector() -> None:
    """The facade runs only aiohttp and ignores any other value."""
    assert "backend" not in inspect.signature(run_one).parameters


@pytest.mark.parametrize("cls", [RunResult, Summary])
def test_results_carry_no_backend_attribution(cls) -> None:
    """A backend field let the report print an aiohttp result as ``httpx``."""
    assert "backend" not in {f.name for f in dataclasses.fields(cls)}


@pytest.mark.asyncio
async def test_run_one_leaves_dyn_http_backend_unset(monkeypatch) -> None:
    """This catches a write to the environment without a ``backend`` parameter.

    The signature test does not see that write.
    """
    monkeypatch.delenv("DYN_HTTP_BACKEND", raising=False)
    result = await run_one([], timeout=1.0, request_rate=100.0)
    assert result.n == 0
    assert "DYN_HTTP_BACKEND" not in os.environ


@pytest.mark.asyncio
async def test_run_one_does_not_clobber_an_operator_set_backend(monkeypatch) -> None:
    monkeypatch.setenv("DYN_HTTP_BACKEND", "aiohttp")
    await run_one([], timeout=1.0, request_rate=100.0)
    assert os.environ["DYN_HTTP_BACKEND"] == "aiohttp"
