# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import struct

import msgspec
import numpy as np
import pytest
import zstandard as zstd

from dynamo.artifacts.format_v1 import (
    CODEC_NONE,
    CODEC_ZSTD,
    GenerationArtifactChoice,
    GenerationArtifactFormatError,
    GenerationArtifactView,
    decode_generation_artifact,
    encode_generation_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

_PRELUDE = struct.Struct("<8sHHHIQ")


def _artifact_view() -> GenerationArtifactView:
    routes = np.array(
        [
            [[0, 2], [1, 3]],
            [[1, 2], [0, 3]],
            [[2, 3], [1, 0]],
            [[3, 1], [2, 0]],
        ],
        dtype=np.int32,
    )
    return GenerationArtifactView(
        choices=(
            GenerationArtifactChoice(
                choice_index=0,
                prompt_token_count=2,
                sequence_token_ids=np.array([101, 102, 201, 202], dtype=np.int64),
                routed_experts=routes,
                router_ids=(3, 7),
                expert_counts=(4, 4),
                selected_logprobs=np.array([-0.25, -0.5], dtype=np.float32),
                selected_logprobs_token_start=2,
            ),
        )
    )


@pytest.mark.parametrize("codec", [CODEC_NONE, CODEC_ZSTD])
def test_generation_artifact_round_trip_is_deterministic(codec: int) -> None:
    encoded = encode_generation_artifact(_artifact_view(), codec=codec)
    assert encoded.data == encode_generation_artifact(_artifact_view(), codec=codec).data
    assert encoded.byte_count == len(encoded.data)
    assert encoded.sha256 == hashlib.sha256(encoded.data).hexdigest()

    decoded = decode_generation_artifact(encoded.data)
    assert decoded.manifest["choices"][0]["prompt_token_count"] == 2
    np.testing.assert_array_equal(
        decoded.choices[0].sequence_token_ids,
        np.array([101, 102, 201, 202], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        decoded.choices[0].routed_experts, _artifact_view().choices[0].routed_experts
    )
    np.testing.assert_allclose(
        decoded.choices[0].selected_logprobs, np.array([-0.25, -0.5], dtype=np.float32)
    )


def test_prelude_and_padding_match_protocol() -> None:
    encoded = encode_generation_artifact(_artifact_view(), codec=CODEC_NONE).data
    magic, major, minor, codec, manifest_bytes, payload_bytes = _PRELUDE.unpack_from(
        encoded
    )
    assert (magic, major, minor, codec) == (b"DYNEXP1\0", 1, 0, CODEC_NONE)

    body = encoded[_PRELUDE.size :]
    manifest = msgspec.msgpack.decode(body[:manifest_bytes])
    payload_start = (manifest_bytes + 63) & ~63
    assert body[manifest_bytes:payload_start] == bytes(payload_start - manifest_bytes)
    assert len(body) == payload_start + payload_bytes
    assert manifest["choices"][0]["sequence_token_ids"]["offset"] == 0


def test_zstd_body_is_one_frame_with_declared_content_size() -> None:
    encoded = encode_generation_artifact(_artifact_view(), codec=CODEC_ZSTD).data
    _, _, _, _, manifest_bytes, payload_bytes = _PRELUDE.unpack_from(encoded)
    compressed = encoded[_PRELUDE.size :]
    params = zstd.get_frame_parameters(compressed)
    expected = ((manifest_bytes + 63) & ~63) + payload_bytes
    assert params.content_size == expected
    assert zstd.ZstdDecompressor().decompress(compressed, max_output_size=expected)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: b"BADMAGIC" + value[8:], "magic"),
        (
            lambda value: value[:8] + struct.pack("<H", 2) + value[10:],
            "major version",
        ),
        (
            lambda value: value[:12] + struct.pack("<H", 99) + value[14:],
            "codec",
        ),
        (lambda value: value + b"trailing", "length"),
    ],
)
def test_decoder_rejects_invalid_container(mutate, message: str) -> None:
    encoded = encode_generation_artifact(_artifact_view(), codec=CODEC_NONE).data
    with pytest.raises(GenerationArtifactFormatError, match=message):
        decode_generation_artifact(mutate(encoded))


def test_encoder_rejects_route_alignment_and_expert_range() -> None:
    choice = _artifact_view().choices[0]
    with pytest.raises(GenerationArtifactFormatError, match="route token count"):
        encode_generation_artifact(
            GenerationArtifactView(
                choices=(
                    GenerationArtifactChoice(
                        choice_index=0,
                        prompt_token_count=2,
                        sequence_token_ids=choice.sequence_token_ids,
                        routed_experts=choice.routed_experts[:3],
                        router_ids=choice.router_ids,
                        expert_counts=choice.expert_counts,
                    ),
                )
            )
        )

    invalid_routes = np.array(choice.routed_experts, copy=True)
    invalid_routes[0, 0, 0] = 4
    with pytest.raises(GenerationArtifactFormatError, match="expert ID"):
        encode_generation_artifact(
            GenerationArtifactView(
                choices=(
                    GenerationArtifactChoice(
                        choice_index=0,
                        prompt_token_count=2,
                        sequence_token_ids=choice.sequence_token_ids,
                        routed_experts=invalid_routes,
                        router_ids=choice.router_ids,
                        expert_counts=choice.expert_counts,
                    ),
                )
            )
        )


def test_encoder_rejects_selected_logprob_misalignment() -> None:
    choice = _artifact_view().choices[0]
    with pytest.raises(GenerationArtifactFormatError, match="selected logprob"):
        encode_generation_artifact(
            GenerationArtifactView(
                choices=(
                    GenerationArtifactChoice(
                        choice_index=0,
                        prompt_token_count=2,
                        sequence_token_ids=choice.sequence_token_ids,
                        selected_logprobs=np.array([-0.1], dtype=np.float32),
                        selected_logprobs_token_start=2,
                    ),
                )
            )
        )
