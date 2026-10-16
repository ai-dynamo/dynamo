#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Regression tests for the --dyn-debug-perf instrumentation module.

#16572: ``SglangProcessor.generator`` imported ``perf_instrumentation`` per
request when ``--dyn-debug-perf`` was on, but the module had been removed, so
every chat request failed with ``ModuleNotFoundError``.
"""

import asyncio

import pytest
from _routed_engine_fakes import FakeRoutedEngine

import dynamo.frontend.sglang_processor as sglang_processor_module
from dynamo.frontend import perf_instrumentation
from dynamo.frontend.sglang_processor import SglangProcessor

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture
def fresh_counters():
    perf_instrumentation._active_requests = 0
    yield
    perf_instrumentation._active_requests = 0


def test_gauge_tracks_concurrency(fresh_counters):
    assert perf_instrumentation.enter_generator() == 1
    assert perf_instrumentation.enter_generator() == 2
    assert perf_instrumentation.exit_generator() == 1
    assert perf_instrumentation.exit_generator() == 0


def test_generator_with_debug_perf_logs_timings(fresh_counters, caplog):
    class OneTokenTokenizer:
        chat_template = ""

        def apply_chat_template(self, messages, **kwargs):
            return [1]

        def decode(self, token_ids, *, skip_special_tokens):
            return "A" * len(token_ids)

    engine = FakeRoutedEngine(items=[{"token_ids": [3], "finish_reason": "stop"}])
    processor = SglangProcessor(
        OneTokenTokenizer(), engine, None, None, None, debug_perf=True
    )
    request = {"model": "test", "messages": [{"role": "user", "content": "Hi"}]}

    async def collect():
        return [item async for item in processor.generator(request)]

    with caplog.at_level("INFO", logger=sglang_processor_module.__name__):
        output = asyncio.run(collect())

    content = "".join(
        choice["delta"].get("content", "")
        for item in output
        for choice in item.get("data", {}).get("choices", [])
    )
    assert content == "A"
    assert "[perf] sglang generator enter: active_requests=1" in caplog.text
    assert "[perf] sglang preprocess:" in caplog.text
    assert "[perf] sglang stream done:" in caplog.text
    exit_lines = [m for m in caplog.messages if "sglang generator exit:" in m]
    assert len(exit_lines) == 1 and exit_lines[0].endswith("active_requests=0")
