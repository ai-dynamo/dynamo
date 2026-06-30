# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit smoke for the client-supplied-embeddings path (PR6).

Proves the path works with no new Dynamo runtime code: a client encodes per-image
embeddings as a safetensors ``data:`` URI, the test-only ``EmbedsPassthroughEncoder``
decodes them (preprocess) and passes them through (forward_batch, identity), and
they splice cleanly via ``build_mixed_embeds``. CPU-only, no model/LM.
"""

import pytest
import torch

from dynamo.vllm.multimodal_utils.async_vision_encoder import AsyncVisionEncoder
from dynamo.vllm.multimodal_utils.embed_assembler import build_mixed_embeds
from tests.utils.embeds_passthrough_encoder import (
    EmbedsPassthroughEncoder,
    decode_embeds_data_uri,
    encode_embeds_data_uri,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]


class _StubNoBuild(EmbedsPassthroughEncoder):
    """Skip the real tokenizer load — the embeds stub only decodes + passes through,
    and image_token_id is hardcoded (151655 via QwenVisionEncoderBackend)."""

    def build(self, model_id):
        pass


def test_encode_decode_roundtrip_preserves_shape_dtype_values():
    t = torch.randn(5, 8, dtype=torch.float32)
    out = decode_embeds_data_uri(encode_embeds_data_uri(t))
    assert out.shape == (5, 8) and out.dtype == torch.float32
    assert torch.equal(out, t)


def test_encode_preserves_bfloat16():
    t = torch.randn(3, 4).to(torch.bfloat16)
    out = decode_embeds_data_uri(encode_embeds_data_uri(t))
    assert out.dtype == torch.bfloat16 and out.shape == (3, 4)


def test_encode_requires_2d():
    with pytest.raises(ValueError, match="2D"):
        encode_embeds_data_uri(torch.randn(8))


def test_decode_rejects_non_data_uri():
    with pytest.raises(ValueError, match="data: URI"):
        decode_embeds_data_uri("https://example.com/img.png")


def test_decode_rejects_malformed_payload():
    with pytest.raises(ValueError, match="no base64 payload"):
        decode_embeds_data_uri("data:application/x-dynamo-embeds;base64,")


def test_image_token_id_is_hardcoded():
    # Inherited from QwenVisionEncoderBackend (Qwen <|image_pad|>), no tokenizer.
    assert EmbedsPassthroughEncoder.image_token_id == 151655


async def test_passthrough_through_async_encoder():
    """encode([data_uri]) decodes the client embedding and returns it unchanged."""
    enc = AsyncVisionEncoder(_StubNoBuild())
    enc.load("fake-model")
    try:
        embeds = torch.randn(7, 6)
        out = await enc.encode([encode_embeds_data_uri(embeds)])
        assert len(out) == 1
        assert torch.equal(out[0], embeds)  # identity passthrough
        assert enc.get_image_placeholder_token_id() == 151655
    finally:
        enc.shutdown()


async def test_bad_embeds_uri_fails_request_atomically():
    """A non-embeds URL fails in preprocess (A5 barrier) — no GPU work, clean error."""
    enc = AsyncVisionEncoder(_StubNoBuild())
    enc.load("fake-model")
    try:
        with pytest.raises(ValueError, match="data: URI"):
            await enc.encode(["http://not-an-embedding"])
    finally:
        enc.shutdown()


def test_decoded_embeds_splice_into_mixed_prompt():
    """The decoded client embeds land at the placeholder rows via build_mixed_embeds."""
    embeds = torch.randn(3, 4)
    decoded = decode_embeds_data_uri(encode_embeds_data_uri(embeds))
    placeholder_id = EmbedsPassthroughEncoder.image_token_id
    token_ids = [10, 11, placeholder_id, 12]
    prompt_embeds, out_ids, is_token_ids = build_mixed_embeds(
        token_ids, [decoded], placeholder_id
    )
    assert out_ids == [10, 11, placeholder_id, placeholder_id, placeholder_id, 12]
    assert is_token_ids == [True, True, False, False, False, True]
    assert torch.equal(prompt_embeds[2:5].to(torch.float32), embeds.to(torch.float32))
