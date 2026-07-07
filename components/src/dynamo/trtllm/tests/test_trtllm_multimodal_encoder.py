# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TRT-LLM multimodal encoder wrapper."""

from unittest import mock

import pytest

from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.engine import TensorRTLLMEngine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.asyncio
async def test_encode_forwards_multimodal_encoder_args():
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
        mock.patch("dynamo.trtllm.engine.MultimodalEncoder") as encoder_cls,
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
