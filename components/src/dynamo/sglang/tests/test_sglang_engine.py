# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``SglangLLMEngine`` — the SGLang LLMEngine implementation
for the unified backend.

Only tests that exercise actual logic — see vllm/tests/test_engine.py for
the rationale.
"""

from __future__ import annotations

import asyncio
import importlib.util
from typing import cast

import pytest
import pytest_asyncio

MODEL_ID = "Qwen/Qwen3-0.6B"
_BASE_ARGV = [
    "--model-path",
    MODEL_ID,
    "--skip-tokenizer-init",
    "--mem-fraction-static",
    "0.3",
    "--context-length",
    "1024",
    "--max-running-requests",
    "4",
    "--disable-cuda-graph",
]

pytestmark = [
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.integration,
    pytest.mark.sglang,
    pytest.mark.unified,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.timeout(300),
    pytest.mark.skipif(
        importlib.util.find_spec("sglang") is None,
        reason="sglang not installed in this container",
    ),
]


class _FakeContext:
    """Duck-typed ``dynamo._core.Context``.  SGLang engine reads
    ``context.trace_id`` (and ``is_stopped()`` in some streaming paths)."""

    def __init__(self, request_id: str = "unit-test-req") -> None:
        self._id = request_id

    @property
    def trace_id(self) -> str:
        return self._id

    def id(self) -> str:
        return self._id

    def is_stopped(self) -> bool:
        return False

    async def async_killed_or_stopped(self) -> None:
        await asyncio.Event().wait()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def started_engine():
    from dynamo.sglang.llm_engine import SglangLLMEngine

    engine, _ = await SglangLLMEngine.from_args(_BASE_ARGV)
    try:
        # Worker_id 0 is fine for tests — the engine doesn't use it.
        engine_config = await engine.start(0)
        yield engine, engine_config
    finally:
        await engine.cleanup()


async def test_start_populates_registration_metadata(started_engine):
    """``start`` must derive ``kv_cache_block_size`` from SGLang's page size
    and populate ``max_num_seqs`` from ``max-running-requests`` — without
    these values the Rust registration path has no routing signals."""
    _engine, cfg = started_engine
    assert cfg.kv_cache_block_size and cfg.kv_cache_block_size > 0
    # total_kv_blocks is derived as ceil(max_total_tokens / page_size).
    # When the engine surfaces max_total_num_tokens, the result must be > 0.
    if cfg.total_kv_blocks is not None:
        assert cfg.total_kv_blocks > 0
    assert cfg.max_num_seqs == 4


async def test_runtime_data_includes_worker_group(monkeypatch):
    from dynamo.sglang import llm_engine as llm_engine_mod
    from dynamo.sglang._disagg import SGLANG_WORKER_GROUP_ID_KEY

    monkeypatch.setattr(
        llm_engine_mod,
        "get_sglang_worker_group_id",
        lambda server_args: "dist_init:tcp://10.0.0.1:2345",
    )

    assert llm_engine_mod._get_runtime_data(object()) == {
        SGLANG_WORKER_GROUP_ID_KEY: "dist_init:tcp://10.0.0.1:2345"
    }


async def test_generate_streams_chunks_with_coherent_final_usage(started_engine):
    engine, _ = started_engine
    ctx = _FakeContext("gen-1")

    chunks = []
    async for chunk in engine.generate(
        cast(
            dict,
            {
                "token_ids": [1, 2, 3, 4, 5],
                "stop_conditions": {"max_tokens": 16},
                "sampling_options": {"temperature": 0.0},
            },
        ),
        cast(object, ctx),  # type: ignore[arg-type]
    ):
        chunks.append(chunk)
        assert "token_ids" in chunk

    final = chunks[-1]
    assert "finish_reason" in final
    usage = final["completion_usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


async def test_abort_and_cleanup_are_safe_before_start():
    from dynamo.sglang.llm_engine import SglangLLMEngine

    engine, _ = await SglangLLMEngine.from_args(_BASE_ARGV)
    await engine.abort(cast(object, _FakeContext()))  # type: ignore[arg-type]
    await engine.cleanup()
    await engine.cleanup()


async def test_abort_unknown_request_on_running_engine(started_engine):
    """SGLang's ``abort_request`` is best-effort; an unknown rid must not raise."""
    engine, _ = started_engine
    await engine.abort(cast(object, _FakeContext("never-submitted")))  # type: ignore[arg-type]
