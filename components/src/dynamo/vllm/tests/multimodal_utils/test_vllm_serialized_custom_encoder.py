# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.vllm.multimodal_utils.serialized_custom_encoder.

SerializedCustomEncoder is the safe default: the subclass writes a synchronous
``_encode_blocking`` and the base runs it via asyncio.to_thread under an
internal lock. These tests pin that encode returns the blocking result, that
concurrent encode calls are serialized (one forward at a time), and that the
class stays abstract.
"""

import asyncio
import time
from typing import List

import pytest
import torch

from dynamo.vllm.multimodal_utils.serialized_custom_encoder import (
    SerializedCustomEncoder,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _CountingEncoder(SerializedCustomEncoder):
    """Records peak concurrency inside _encode_blocking to prove serialization."""

    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0

    def load(self, model_id: str, device: str) -> None:
        ...

    def get_image_placeholder_token_id(self) -> int:
        return 1

    def _encode_blocking(self, image_urls: List[str]) -> List[torch.Tensor]:
        # Serialized by the base, so only one thread is ever in here; a plain
        # (non-atomic) counter is sufficient and its peak must stay 1.
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        time.sleep(0.02)  # widen the window so a missing lock would overlap
        self.active -= 1
        return [torch.zeros(1, 4) for _ in image_urls]


async def test_encode_returns_blocking_result():
    enc = _CountingEncoder()
    out = await enc.encode(["a", "b"])
    assert len(out) == 2
    assert all(t.shape == (1, 4) for t in out)


async def test_encode_serializes_concurrent_calls():
    """Concurrent encode calls never run _encode_blocking simultaneously."""
    enc = _CountingEncoder()
    await asyncio.gather(*[enc.encode(["u"]) for _ in range(5)])
    assert enc.max_active == 1


def test_encode_blocking_is_abstract():
    """A subclass that omits _encode_blocking cannot instantiate."""

    class _NoBlocking(SerializedCustomEncoder):
        def load(self, model_id: str, device: str) -> None:
            ...

        def get_image_placeholder_token_id(self) -> int:
            return 1

    with pytest.raises(TypeError):
        _NoBlocking()


def test_serialized_custom_encoder_is_abstract():
    """load / get_image_placeholder_token_id / _encode_blocking stay abstract."""
    with pytest.raises(TypeError):
        SerializedCustomEncoder()
