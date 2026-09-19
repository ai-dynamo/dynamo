# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests: the sweep harness must not attribute a result to a backend.

The harness used to pick a backend by writing ``DYN_HTTP_BACKEND`` and then
label the result with the string it was handed. Once the facade began warning
and falling back to aiohttp for any other value, a run requested as ``httpx``
was measured by ``AiohttpClient`` and still printed under an ``httpx`` column —
a clean-looking A/B table in which both sides were aiohttp. There is now one
backend, so there is no selector to pass and no label to be wrong. These pin
that.
"""

from __future__ import annotations

import dataclasses
import inspect
import os

import pytest

from benchmarks.multimodal.http.runner import RunResult, run_one
from benchmarks.multimodal.http.stats import Summary

# Deliberately unmarked: the repository-root conftest auto-applies a lifecycle
# plus ``defaulted``, which is what routes a benchmarks/ test into the CPU
# parallel job. Hand-marking is what keeps benchmarks/router/tests/test_common.py
# (pre_merge/gpu_0/unit/parallel) out of every job in CI today.


def test_run_one_takes_no_backend_selector() -> None:
    """No caller can ask the harness for a backend it cannot deliver."""
    assert "backend" not in inspect.signature(run_one).parameters


@pytest.mark.parametrize("cls", [RunResult, Summary])
def test_results_carry_no_backend_attribution(cls) -> None:
    """Nothing downstream can print a result under a backend name."""
    assert "backend" not in {f.name for f in dataclasses.fields(cls)}


@pytest.mark.asyncio
async def test_run_one_leaves_dyn_http_backend_unset(monkeypatch) -> None:
    """A run must not introduce the deprecated selector into the environment."""
    monkeypatch.delenv("DYN_HTTP_BACKEND", raising=False)
    result = await run_one([], timeout=1.0, request_rate=100.0)
    assert result.n == 0
    assert "DYN_HTTP_BACKEND" not in os.environ


@pytest.mark.asyncio
async def test_run_one_does_not_clobber_an_operator_set_backend(monkeypatch) -> None:
    """A run must leave an operator's own DYN_HTTP_BACKEND value alone."""
    monkeypatch.setenv("DYN_HTTP_BACKEND", "aiohttp")
    await run_one([], timeout=1.0, request_rate=100.0)
    assert os.environ["DYN_HTTP_BACKEND"] == "aiohttp"
