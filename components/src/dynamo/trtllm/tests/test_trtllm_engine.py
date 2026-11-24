# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TRTLLM engine wrapper."""

import asyncio
import threading
from unittest import mock

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )
import tensorrt_llm.llmapi.llm as trtllm_llm_module

from dynamo.trtllm.engine import TensorRTLLMEngine, get_llm_engine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    # NOTE: these tests use no GPU, but importing tensorrt_llm needs one, and the
    # gpu_0 stage also runs on GPU-less runners.
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]

# Intentionally unprofiled: these import-heavy, zero-VRAM tests run in the
# sequential GPU stage so TensorRT-LLM initialization is shared.

_PYTORCH_LLM_CLS_NAME = "dynamo.trtllm.engine.LLM"


class TestTensorRTLLMEngine:
    @pytest.mark.parametrize("backend", ["foo", "bar", "cpp", "_autodeploy"])
    def test_raises_on_unsupported_backends(self, backend):
        with pytest.raises(ValueError, match="Unsupported backend"):
            TensorRTLLMEngine(engine_args={"backend": backend})

    @pytest.mark.asyncio
    async def test_picks_expected_llm_cls(self):
        with mock.patch(_PYTORCH_LLM_CLS_NAME) as mocked_cls:
            engine = TensorRTLLMEngine(engine_args={"backend": "pytorch"})
            await engine.initialize()

        mocked_cls.assert_called_once()

    def test_get_kv_cache_capacity_delegates_to_llm(self):
        capacity = {
            "maxNumBlocks": 123,
            "tokensPerBlock": 64,
            "maxNumTokens": 7872,
        }
        engine = TensorRTLLMEngine(engine_args={})
        engine._llm = mock.Mock()
        engine._llm.get_kv_cache_capacity.return_value = capacity

        assert engine.get_kv_cache_capacity() == capacity
        engine._llm.get_kv_cache_capacity.assert_called_once_with()

    def test_get_kv_cache_capacity_returns_empty_when_api_is_unavailable(self):
        engine = TensorRTLLMEngine(engine_args={})
        engine._llm = mock.Mock(spec=[])

        assert engine.get_kv_cache_capacity() == {}


@pytest.mark.asyncio
async def test_get_llm_engine_forwards_backend():
    engine_args = {"foo": mock.Mock(), "backend": "pytorch"}
    with mock.patch(
        "dynamo.trtllm.engine.TensorRTLLMEngine", return_value=mock.AsyncMock()
    ) as mocked_engine:
        async with get_llm_engine(engine_args=engine_args):
            pass

    mocked_engine.assert_called_once_with(engine_args, None)
    mocked_engine.return_value.start_preprocess_pool.assert_not_awaited()


@pytest.mark.multimodal
@pytest.mark.asyncio
async def test_get_llm_engine_runs_a_preprocess_pool_on_request():
    calls = mock.Mock()
    with (
        mock.patch(_PYTORCH_LLM_CLS_NAME, return_value=calls.llm),
        mock.patch(
            "dynamo.trtllm.engine.PreprocessPool.start", return_value=calls.pool
        ) as start,
    ):
        async with get_llm_engine(engine_args={}, mm_preprocess_processes=4):
            start.assert_called_once_with(trtllm_llm_module, calls.llm, 4)

    # The pool closes before the LLM shuts down.
    assert calls.mock_calls == [mock.call.pool.close(), mock.call.llm.shutdown()]


@pytest.fixture
def engine():
    """An engine with a mock LLM, shut down after the test."""
    engine = TensorRTLLMEngine(engine_args={})
    engine._llm = mock.Mock()
    yield engine
    engine.shutdown()


@pytest.mark.multimodal
class TestPreprocess:
    @pytest.mark.asyncio
    async def test_preprocesses_one_request_per_pool_process(self, engine):
        with mock.patch("dynamo.trtllm.engine.PreprocessPool.start"):
            await engine.start_preprocess_pool(3)
        # Each request waits until all three are being preprocessed at once.
        barrier = threading.Barrier(3, timeout=10)

        def preprocess(inputs, *args):
            barrier.wait()
            return inputs

        engine._llm.preprocess.side_effect = preprocess
        results = await asyncio.gather(
            *(engine.preprocess(i, None, None) for i in range(3))
        )
        assert results == [0, 1, 2]

    @pytest.mark.asyncio
    async def test_drops_a_request_cancelled_while_it_waits_for_the_thread(
        self, engine
    ):
        started, finish = threading.Event(), threading.Event()

        def preprocess(inputs, *args):
            started.set()
            assert finish.wait(timeout=10)
            return inputs

        engine._llm.preprocess.side_effect = preprocess
        first = asyncio.create_task(engine.preprocess("first", None, None))
        assert await asyncio.to_thread(started.wait, 10)
        # Without a pool, the second request waits for the first one's thread.
        second = asyncio.create_task(engine.preprocess("second", None, None))
        await asyncio.sleep(0)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second

        finish.set()
        assert await first == "first"
        engine._llm.preprocess.assert_called_once_with("first", None, None)

    @pytest.mark.asyncio
    async def test_releases_the_inputs_of_a_request_cancelled_while_preprocessing(
        self, engine
    ):
        started, finish, released = (threading.Event() for _ in range(3))
        preprocessed = mock.Mock()
        preprocessed.multimodal_params.release_handles.side_effect = released.set

        def preprocess(*args):
            started.set()
            assert finish.wait(timeout=10)
            return preprocessed

        engine._llm.preprocess.side_effect = preprocess
        task = asyncio.create_task(engine.preprocess("inputs", None, None))
        assert await asyncio.to_thread(started.wait, 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Preprocessing finishes after the request was cancelled.
        finish.set()
        assert await asyncio.to_thread(released.wait, 10)
