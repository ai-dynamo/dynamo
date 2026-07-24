# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin TRT-LLM's MM-routing wire-protocol contracts against drift.

These fast unit tests import the installed tensorrt_llm module and verify the
contracts Dynamo depends on for TRT-LLM MM-aware KV routing: Qwen image markers
remain in-vocabulary, and request prompts carry ``multi_modal_uuids``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "CUDA/GPU unavailable; importing TensorRT-LLM requires CUDA.",
        allow_module_level=True,
    )

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.trtllm,
    pytest.mark.multimodal,
    pytest.mark.unit,
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(0),
]


def test_trtllm_qwen2vl_uses_in_vocab_image_markers() -> None:
    from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase

    processor = object.__new__(Qwen2VLInputProcessorBase)
    processor._config = SimpleNamespace(
        image_token_id=91,
        vision_token_id=92,
        video_token_id=93,
        text_config=SimpleNamespace(vocab_size=100),
    )
    input_ids = torch.tensor([1, 91, 91, 2], dtype=torch.int32)

    assert torch.equal(processor._postprocess(input_ids), input_ids)
    assert processor.get_mm_token_ids().tolist() == [91, 92, 93]
    assert all(token_id < processor.get_vocab_size() for token_id in [91, 92, 93])


def test_trtllm_multi_modal_uuids_field_present() -> None:
    from tensorrt_llm.inputs.data import TextPrompt, TokensPrompt

    for cls in (TextPrompt, TokensPrompt):
        assert "multi_modal_uuids" in cls.__annotations__, (
            f"tensorrt_llm.inputs.data.{cls.__name__}.multi_modal_uuids is gone; "
            "dynamo forwards the frontend's mm_hash through it "
            "(components/src/dynamo/trtllm/multimodal_processor.py) so TRT-LLM "
            "echoes the routing hash in its KV events. An upstream rename breaks "
            "MM-aware routing."
        )
