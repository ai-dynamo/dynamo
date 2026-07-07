# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TRT-LLM multimodal encoder wrapper."""

import importlib
import sys
from types import ModuleType
from unittest import mock

import pytest

from dynamo.trtllm.constants import DisaggregationMode

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture
def trtllm_engine_module(monkeypatch):
    """Import the engine without loading TRT-LLM's CUDA-linked libraries."""

    class _FakeLLM:
        pass

    class _FakeMultimodalEncoder:
        pass

    class _FakeBaseLLM:
        pass

    class _FakeAutoConfig:
        pass

    trtllm_module = ModuleType("tensorrt_llm")
    trtllm_module.__path__ = []  # type: ignore[attr-defined]
    trtllm_module.LLM = _FakeLLM  # type: ignore[attr-defined]
    trtllm_module.MultimodalEncoder = _FakeMultimodalEncoder  # type: ignore[attr-defined]

    llmapi_module = ModuleType("tensorrt_llm.llmapi")
    llmapi_module.__path__ = []  # type: ignore[attr-defined]
    llm_module = ModuleType("tensorrt_llm.llmapi.llm")
    llm_module.BaseLLM = _FakeBaseLLM  # type: ignore[attr-defined]
    transformers_module = ModuleType("transformers")
    transformers_module.AutoConfig = _FakeAutoConfig  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "tensorrt_llm", trtllm_module)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", llmapi_module)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi.llm", llm_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)

    module_name = "dynamo.trtllm.engine"
    existing_module = sys.modules.pop(module_name, None)
    try:
        yield importlib.import_module(module_name)
    finally:
        sys.modules.pop(module_name, None)
        if existing_module is not None:
            sys.modules[module_name] = existing_module


@pytest.mark.asyncio
async def test_encode_forwards_multimodal_encoder_args(trtllm_engine_module):
    TensorRTLLMEngine = trtllm_engine_module.TensorRTLLMEngine
    model_kwargs = {
        "torch_dtype": "bfloat16",
        "text_config": {"torch_dtype": "bfloat16"},
    }
    engine_args = {
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "max_batch_size": 8,
        "trust_remote_code": True,
        "tensor_parallel_size": 2,
        "model_kwargs": model_kwargs,
        "kv_cache_config": {"free_gpu_memory_fraction": 0.8},
    }

    with (
        mock.patch.object(
            TensorRTLLMEngine,
            "_is_unsupported_encoder_arch",
            return_value=False,
        ),
        mock.patch.object(trtllm_engine_module, "MultimodalEncoder") as encoder_cls,
    ):
        engine = TensorRTLLMEngine(engine_args, DisaggregationMode.ENCODE)
        await engine.initialize()

    encoder_cls.assert_called_once_with(
        model="Qwen/Qwen3-VL-2B-Instruct",
        max_batch_size=8,
        trust_remote_code=True,
        tensor_parallel_size=2,
        model_kwargs=model_kwargs,
    )
