# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ``VllmLLMEngine``.

These exercise the adapter logic between the ``LLMEngine`` ABC and vLLM's
``AsyncLLM`` using a stub ``AsyncLLM``. They do NOT spawn a real vLLM
``EngineCore`` subprocess.

End-to-end coverage of a real vLLM engine lifecycle (start, generate,
shutdown) runs through
``tests/serve/test_vllm.py::test_serve_deployment[aggregated_unified]``,
which boots a real vllm-backed ``dynamo serve`` in a subprocess and issues
chat / completion requests. That is the right place to assert "vLLM
actually works" — pytest-process spawning of vLLM here adds no extra
coverage and re-imports ``vllm`` from a dirty ``sys.path`` in the child.
"""

from __future__ import annotations

import asyncio
import importlib.util
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    # gpu_1 not gpu_0: importing dynamo.vllm.handlers pulls in vllm modules
    # whose import-time setup fails on CPU-only arm64. Mirrors test_vllm_unit.py.
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
    pytest.mark.skipif(
        importlib.util.find_spec("vllm") is None,
        reason="vllm not installed in this container",
    ),
]


class _FakeContext:
    """Duck-typed ``dynamo._core.Context``. ``VllmLLMEngine`` only calls
    ``context.id()``."""

    def __init__(self, request_id: str = "unit-test-req") -> None:
        self._id = request_id

    def id(self) -> str:
        return self._id


def _fake_vllm_config(
    *,
    max_model_len: int = 1024,
    block_size: int = 16,
    num_gpu_blocks: int = 128,
    max_num_seqs: int = 4,
    max_num_batched_tokens: int = 2048,
) -> SimpleNamespace:
    """Stub the subset of ``VllmConfig`` that ``VllmLLMEngine.start`` reads."""
    return SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_model_len),
        cache_config=SimpleNamespace(
            block_size=block_size, num_gpu_blocks=num_gpu_blocks
        ),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
        ),
    )


def _make_engine(vllm_config: SimpleNamespace | None = None):
    """Construct a ``VllmLLMEngine`` directly with stubbed ``engine_args``,
    bypassing ``parse_args`` so the test doesn't need a real vLLM CLI parse."""
    from dynamo.vllm.llm_engine import VllmLLMEngine

    if vllm_config is None:
        vllm_config = _fake_vllm_config()

    engine_args = SimpleNamespace(
        model="Qwen/Qwen3-0.6B",
        served_model_name="Qwen/Qwen3-0.6B",
        create_engine_config=MagicMock(return_value=vllm_config),
        create_model_config=MagicMock(
            return_value=SimpleNamespace(
                get_diff_sampling_param=MagicMock(return_value={})
            )
        ),
    )
    return VllmLLMEngine(engine_args)


@pytest.mark.asyncio
async def test_start_populates_registration_metadata():
    """``start`` must surface non-None values pulled from ``VllmConfig`` —
    the Rust registration path reads these fields, and a None there means
    the model appears in /v1/models but fails to actually serve."""
    vllm_config = _fake_vllm_config(
        max_model_len=2048,
        block_size=32,
        num_gpu_blocks=256,
        max_num_seqs=8,
        max_num_batched_tokens=4096,
    )
    engine = _make_engine(vllm_config=vllm_config)

    fake_async_llm = MagicMock()
    with patch(
        "dynamo.vllm.llm_engine.AsyncLLM.from_vllm_config",
        return_value=fake_async_llm,
    ):
        cfg = await engine.start()

    assert cfg.context_length == 2048
    assert cfg.kv_cache_block_size == 32
    assert cfg.total_kv_blocks == 256
    assert cfg.max_num_seqs == 8
    assert cfg.max_num_batched_tokens == 4096
    assert cfg.model == "Qwen/Qwen3-0.6B"
    assert cfg.served_model_name == "Qwen/Qwen3-0.6B"

    await engine.cleanup()


@pytest.mark.asyncio
async def test_generate_streams_chunks_with_coherent_final_usage():
    """Every chunk carries ``token_ids``; the final chunk additionally carries
    ``finish_reason`` and a ``completion_usage`` whose totals add up."""
    engine = _make_engine()

    def _output(token_ids, finish_reason=None, index=0):
        return SimpleNamespace(
            index=index, token_ids=token_ids, finish_reason=finish_reason
        )

    def _request_output(prompt_token_ids, outputs):
        return SimpleNamespace(prompt_token_ids=prompt_token_ids, outputs=outputs)

    async def fake_generate(prompt, sampling_params, request_id):
        # Two streaming RequestOutputs: cumulative token_ids per vLLM contract,
        # final carries finish_reason.
        yield _request_output([1, 2, 3], [_output([10, 11, 12])])
        yield _request_output(
            [1, 2, 3],
            [_output([10, 11, 12, 13, 14], finish_reason="stop")],
        )

    fake_async_llm = MagicMock()
    fake_async_llm.generate = fake_generate

    with patch(
        "dynamo.vllm.llm_engine.AsyncLLM.from_vllm_config",
        return_value=fake_async_llm,
    ), patch(
        "dynamo.vllm.llm_engine.build_sampling_params",
        return_value=MagicMock(),
    ):
        await engine.start()

        ctx = _FakeContext("gen-1")
        chunks = []
        async for chunk in engine.generate(
            cast(
                dict,
                {
                    "token_ids": [1, 2, 3],
                    "sampling_options": {"temperature": 0.0},
                    "stop_conditions": {"max_tokens": 16},
                },
            ),
            cast(object, ctx),  # type: ignore[arg-type]
        ):
            assert "token_ids" in chunk
            chunks.append(chunk)

    assert len(chunks) == 2
    # First chunk: 3 new tokens, no finish_reason
    assert chunks[0]["token_ids"] == [10, 11, 12]
    assert "finish_reason" not in chunks[0]
    # Final chunk: 2 new tokens (delta from cumulative), finish_reason + usage
    assert chunks[-1]["token_ids"] == [13, 14]
    assert chunks[-1]["finish_reason"] == "stop"
    usage = chunks[-1]["completion_usage"]
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 8

    await engine.cleanup()


@pytest.mark.asyncio
async def test_abort_and_cleanup_are_safe_before_start():
    """Worker may call ``abort`` / ``cleanup`` on any failure path. Neither
    must raise on a just-constructed engine, and ``cleanup`` must be
    idempotent."""
    engine = _make_engine()
    await engine.abort(cast(object, _FakeContext()))  # type: ignore[arg-type]
    await engine.cleanup()
    await engine.cleanup()


@pytest.mark.asyncio
async def test_abort_forwards_request_id_to_engine_client():
    """``abort`` is a thin pass-through to ``AsyncLLM.abort`` keyed by
    ``context.id()``."""
    engine = _make_engine()

    fake_async_llm = MagicMock()

    abort_calls: list[str] = []

    async def fake_abort(request_id: str) -> None:
        abort_calls.append(request_id)

    fake_async_llm.abort = fake_abort

    with patch(
        "dynamo.vllm.llm_engine.AsyncLLM.from_vllm_config",
        return_value=fake_async_llm,
    ):
        await engine.start()
        await engine.abort(cast(object, _FakeContext("req-42")))  # type: ignore[arg-type]

    assert abort_calls == ["req-42"]
    await engine.cleanup()
