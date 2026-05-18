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
import os
from collections.abc import Mapping
from typing import cast

import pytest
import pytest_asyncio

from tests.utils.gpu_args import build_gpu_mem_args

MODEL_ID = "Qwen/Qwen3-0.6B"


def _env_with_requested_kv_bytes(request: pytest.FixtureRequest) -> dict[str, str]:
    env = os.environ.copy()
    if "_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES" not in env:
        kv_mark = request.node.get_closest_marker("requested_vllm_kv_cache_bytes")
        if kv_mark:
            env["_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES"] = str(int(kv_mark.args[0]))
    return env


def _base_argv(env: Mapping[str, str]) -> list[str]:
    gpu_mem_args = build_gpu_mem_args("build_vllm_gpu_mem_args", env=env)
    if not gpu_mem_args:
        gpu_mem_args = ["--gpu-memory-utilization", "0.3"]

    return [
        "--model",
        MODEL_ID,
        *gpu_mem_args,
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


@pytest_asyncio.fixture(loop_scope="module")
async def started_engine(request):
    # Use vLLM's default spawn for the EngineCore subprocess (set by
    # VllmLLMEngine.start). Fork would inherit pytest's filterwarnings=error
    # and crash the child on transitive flashinfer DeprecationWarnings.
    from dynamo.vllm.llm_engine import VllmLLMEngine

    engine, _ = await VllmLLMEngine.from_args(
        _base_argv(_env_with_requested_kv_bytes(request))
    )
    try:
        # Worker_id 0 is fine for tests — the engine doesn't use it.
        engine_config = await engine.start(0)
        yield engine, engine_config
    finally:
        await engine.cleanup()


@pytest.mark.pre_merge
@pytest.mark.profiled_vram_gib(3.8)
@pytest.mark.requested_vllm_kv_cache_bytes(1_119_388_000)
@pytest.mark.timeout(900)
async def test_vllm_engine_all(request, started_engine):
    """Run the real-engine VllmLLMEngine checks under one profiled startup."""
    await _check_start_populates_registration_metadata(started_engine)
    await _check_generate_streams_chunks_with_coherent_final_usage(started_engine)
    await _check_abort_and_cleanup_are_safe_before_start(request)
    await _check_abort_unknown_request_on_running_engine(started_engine)
    await _check_from_args_propagates_disaggregation_mode_to_worker_config(
        request, "agg", "AGGREGATED"
    )
    await _check_from_args_propagates_disaggregation_mode_to_worker_config(
        request, "prefill", "PREFILL"
    )
    await _check_from_args_propagates_disaggregation_mode_to_worker_config(
        request, "decode", "DECODE"
    )


async def _check_start_populates_registration_metadata(started_engine):
    """``start`` must surface non-None values for the fields the Rust
    registration path reads — if any of them come back None, the model
    appears in /v1/models but fails to actually serve."""
    _engine, cfg = started_engine
    assert cfg.context_length and cfg.context_length > 0
    assert cfg.kv_cache_block_size and cfg.kv_cache_block_size > 0
    assert cfg.total_kv_blocks and cfg.total_kv_blocks > 0
    assert cfg.max_num_seqs and cfg.max_num_seqs > 0
    assert cfg.max_num_batched_tokens and cfg.max_num_batched_tokens > 0


async def _check_generate_streams_chunks_with_coherent_final_usage(started_engine):
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


async def _check_abort_and_cleanup_are_safe_before_start(request):
    """Worker may call ``abort`` / ``cleanup`` on any failure path.  Neither
    must raise on a just-constructed engine, and ``cleanup`` must be
    idempotent."""
    from dynamo.vllm.llm_engine import VllmLLMEngine

    engine, _ = await VllmLLMEngine.from_args(
        _base_argv(_env_with_requested_kv_bytes(request))
    )
    await engine.abort(cast(object, _FakeContext()))  # type: ignore[arg-type]
    await engine.cleanup()
    await engine.cleanup()


async def _check_abort_unknown_request_on_running_engine(started_engine):
    """vLLM's ``AsyncLLM.abort`` is a best-effort lookup.  Aborting a request
    id the engine has never seen must not raise."""
    engine, _ = started_engine
    await engine.abort(cast(object, _FakeContext("never-submitted")))  # type: ignore[arg-type]


async def _check_from_args_propagates_disaggregation_mode_to_worker_config(
    request, mode_arg, expected
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
        [
            *_base_argv(_env_with_requested_kv_bytes(request)),
            "--disaggregation-mode",
            mode_arg,
            *extra_args,
        ]
    )
    try:
        expected_mode = getattr(DisaggregationMode, expected)
        assert engine.disaggregation_mode is expected_mode
        assert worker_config.disaggregation_mode is expected_mode
    finally:
        await engine.cleanup()
