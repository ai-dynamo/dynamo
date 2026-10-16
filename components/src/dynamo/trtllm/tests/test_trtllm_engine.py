# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TRTLLM engine wrapper."""

from unittest import mock

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )
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
