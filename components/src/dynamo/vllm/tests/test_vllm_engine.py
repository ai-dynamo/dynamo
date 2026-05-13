# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``VllmLLMEngine`` — the vLLM LLMEngine implementation for
the unified backend.

Only tests that exercise actual logic:
  * ``start`` extracts context_length / kv_cache_block_size / etc. from the
    underlying vLLM engine into ``EngineConfig`` (Rust-side model
    registration depends on these values being populated, not None).
  * ``generate`` produces the streaming chunk shape the Rust layer expects:
    every chunk has ``token_ids``, final chunk adds ``finish_reason`` and a
    coherent ``completion_usage``.
  * ``abort`` and ``cleanup`` are safe to call before ``start`` and on
    already-cleaned engines (Worker may call them on any failure path).
"""

from __future__ import annotations

import asyncio
import importlib.util
from typing import cast

import pytest
import pytest_asyncio

MODEL_ID = "Qwen/Qwen3-0.6B"
_BASE_ARGV = [
    "--model",
    MODEL_ID,
    "--gpu-memory-utilization",
    "0.3",
    "--max-model-len",
    "1024",
    "--max-num-seqs",
    "4",
    "--enforce-eager",
]

pytestmark = [
    pytest.mark.asyncio(loop_scope="module"),
    pytest.mark.integration,
    pytest.mark.vllm,
    pytest.mark.unified,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.timeout(300),
    pytest.mark.skipif(
        importlib.util.find_spec("vllm") is None,
        reason="vllm not installed in this container",
    ),
]


class _FakeContext:
    """Duck-typed ``dynamo._core.Context``.  ``VllmLLMEngine`` only calls
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
    # Use vLLM's default spawn for the EngineCore subprocess (set by
    # VllmLLMEngine.start). Fork would inherit pytest's filterwarnings=error
    # and crash the child on transitive flashinfer DeprecationWarnings.
    from dynamo.vllm.llm_engine import VllmLLMEngine

    engine, _ = await VllmLLMEngine.from_args(_BASE_ARGV)
    try:
        # Worker_id 0 is fine for tests — the engine doesn't use it.
        engine_config = await engine.start(0)
        yield engine, engine_config
    finally:
        await engine.cleanup()


async def test_start_populates_registration_metadata(started_engine):
    """``start`` must surface non-None values for the fields the Rust
    registration path reads — if any of them come back None, the model
    appears in /v1/models but fails to actually serve."""
    _engine, cfg = started_engine
    assert cfg.context_length and cfg.context_length > 0
    assert cfg.kv_cache_block_size and cfg.kv_cache_block_size > 0
    assert cfg.total_kv_blocks and cfg.total_kv_blocks > 0
    assert cfg.max_num_seqs and cfg.max_num_seqs > 0
    assert cfg.max_num_batched_tokens and cfg.max_num_batched_tokens > 0


async def test_generate_streams_chunks_with_coherent_final_usage(started_engine):
    """Every chunk must carry ``token_ids``; the final chunk must also carry
    ``finish_reason`` and a ``completion_usage`` whose totals add up."""
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
    """Worker may call ``abort`` / ``cleanup`` on any failure path.  Neither
    must raise on a just-constructed engine, and ``cleanup`` must be
    idempotent."""
    from dynamo.vllm.llm_engine import VllmLLMEngine

    engine, _ = await VllmLLMEngine.from_args(_BASE_ARGV)
    await engine.abort(cast(object, _FakeContext()))  # type: ignore[arg-type]
    await engine.cleanup()
    await engine.cleanup()


async def test_abort_unknown_request_on_running_engine(started_engine):
    """vLLM's ``AsyncLLM.abort`` is a best-effort lookup.  Aborting a request
    id the engine has never seen must not raise."""
    engine, _ = started_engine
    await engine.abort(cast(object, _FakeContext("never-submitted")))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "mode_arg, expected",
    [
        ("agg", "AGGREGATED"),
        ("prefill", "PREFILL"),
        ("decode", "DECODE"),
    ],
)
async def test_from_args_propagates_disaggregation_mode_to_worker_config(
    mode_arg, expected
):
    """``--disaggregation-mode`` must flow from CLI through to the
    ``WorkerConfig`` the unified Worker sees, and onto the engine instance
    so ``generate()`` can branch on it. Without this hookup the prefill
    role would silently degrade to aggregated."""
    from dynamo.common.constants import DisaggregationMode
    from dynamo.vllm.llm_engine import VllmLLMEngine

    extra_args: list[str] = []
    # vLLM rejects --disaggregation-mode prefill without an explicit
    # --kv-transfer-config; supply NixlConnector to satisfy the validator.
    if mode_arg == "prefill":
        extra_args = [
            "--kv-transfer-config",
            '{"kv_connector":"NixlConnector","kv_role":"kv_both"}',
        ]

    engine, worker_config = await VllmLLMEngine.from_args(
        [*_BASE_ARGV, "--disaggregation-mode", mode_arg, *extra_args]
    )
    try:
        expected_mode = getattr(DisaggregationMode, expected)
        assert engine.disaggregation_mode is expected_mode
        assert worker_config.disaggregation_mode is expected_mode
    finally:
        await engine.cleanup()
