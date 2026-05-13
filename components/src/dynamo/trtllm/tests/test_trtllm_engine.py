# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``TrtllmLLMEngine`` — the TensorRT-LLM LLMEngine
implementation for the unified backend.

Only tests that exercise actual logic — see vllm/tests/test_engine.py for
the rationale.  The lower-level ``TensorRTLLMEngine`` wrapper in
``engine.py`` is intentionally out of scope; it is exercised indirectly
via ``TrtllmLLMEngine.start()``.
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
    "--free-gpu-memory-fraction",
    "0.3",
    "--max-seq-len",
    "1024",
    "--max-batch-size",
    "4",
]

pytestmark = [
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.integration,
    pytest.mark.trtllm,
    pytest.mark.unified,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.timeout(300),
    pytest.mark.skipif(
        importlib.util.find_spec("tensorrt_llm") is None,
        reason="tensorrt_llm not installed in this container",
    ),
]


class _FakeContext:
    """Duck-typed ``dynamo._core.Context``.  TrtllmLLMEngine only calls
    ``context.id()``."""

    def __init__(self, request_id: str = "unit-test-req") -> None:
        self._id = request_id

    def id(self) -> str:
        return self._id

    def is_stopped(self) -> bool:
        return False

    async def async_killed_or_stopped(self) -> None:
        await asyncio.Event().wait()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def started_engine():
    from dynamo.trtllm.llm_engine import TrtllmLLMEngine

    engine, _ = await TrtllmLLMEngine.from_args(_BASE_ARGV)
    try:
        # Worker_id 0 is fine for tests — `disagg_machine_id` is derived
        # from worker_id but unused outside disagg PREFILL/DECODE roles.
        engine_config = await engine.start(0)
        yield engine, engine_config
    finally:
        await engine.cleanup()


async def test_start_populates_registration_metadata(started_engine):
    """``start`` threads ``max_seq_len`` / ``kv_block_size`` / ``max_batch_size``
    from ``from_args``'s parsed config through to ``EngineConfig`` —
    mismatches here surface as incorrect Rust-side routing decisions."""
    _engine, cfg = started_engine
    assert cfg.context_length == 1024
    assert cfg.kv_cache_block_size > 0
    assert cfg.max_num_seqs == 4


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
    from dynamo.trtllm.llm_engine import TrtllmLLMEngine

    engine, _ = await TrtllmLLMEngine.from_args(_BASE_ARGV)
    await engine.abort(cast(object, _FakeContext()))  # type: ignore[arg-type]
    await engine.cleanup()
    await engine.cleanup()


async def test_abort_unknown_request_on_running_engine(started_engine):
    """``_active_requests`` is only populated during ``generate``.  An unknown
    request id is silently ignored rather than raised."""
    engine, _ = started_engine
    await engine.abort(cast(object, _FakeContext("never-submitted")))  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Drain unit tests — exercised on the engine instance directly without
# starting a real engine. The fixture stubs `engine.llm.get_stats_async`
# so we don't need a GPU to assert the drain contract.
# ----------------------------------------------------------------------


class _StubLLM:
    """Minimal ``engine.llm`` stand-in. Each ``get_stats_async()`` call
    returns the next pre-canned stat dict from a queue."""

    def __init__(self, stats_sequence: list[dict]) -> None:
        self._stats = list(stats_sequence)

    def get_stats_async(self, timeout: float):  # noqa: ARG002
        async def _gen():
            if not self._stats:
                # No more stats: pretend the engine timed out the poll.
                # The drain loop swallows this and re-polls.
                raise RuntimeError("no stats")
            yield self._stats.pop(0)

        return _gen()


class _StubInner:
    """Stand-in for the wrapped TensorRTLLMEngine — exposes only `llm`."""

    def __init__(self, llm: _StubLLM) -> None:
        self.llm = llm


def _engine_with_stub(disagg_mode, stats_sequence: list[dict]):
    """Bypass __init__ so we don't need real TRT-LLM state."""
    from dynamo.trtllm.llm_engine import TrtllmLLMEngine

    engine = TrtllmLLMEngine.__new__(TrtllmLLMEngine)
    engine.disaggregation_mode = disagg_mode
    engine._engine = _StubInner(_StubLLM(stats_sequence))
    return engine


async def test_drain_is_noop_for_aggregated_workers():
    """Drain only matters on prefill workers (in-flight NIXL transfers).
    Aggregated/decode workers exit immediately so shutdown isn't gated
    on an unnecessary stats poll."""
    from dynamo.trtllm.constants import DisaggregationMode as TrtDisagg

    engine = _engine_with_stub(TrtDisagg.AGGREGATED, [{"numActiveRequests": 5}])
    await engine.drain()  # must return without consuming a stat


async def test_drain_returns_when_engine_idle():
    """Polls until active+queued == 0, then returns. The drain loop must
    not hang once the engine reports idle."""
    from dynamo.trtllm.constants import DisaggregationMode as TrtDisagg

    engine = _engine_with_stub(
        TrtDisagg.PREFILL,
        [
            {"numActiveRequests": 2, "numQueuedRequests": 1},
            {"numActiveRequests": 0, "numQueuedRequests": 0},
        ],
    )
    await engine.drain()
