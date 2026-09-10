# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import struct
from unittest.mock import patch

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


def _replace_manifest(encoded: bytes, manifest: dict) -> bytes:
    (
        magic,
        major,
        minor,
        codec,
        old_manifest_bytes,
        payload_bytes,
    ) = _PRELUDE.unpack_from(encoded)
    assert codec == CODEC_NONE
    old_body = encoded[_PRELUDE.size :]
    payload = old_body[((old_manifest_bytes + 63) & ~63) :]
    manifest_data = msgspec.msgpack.encode(manifest, order="deterministic")
    body = (
        manifest_data
        + bytes(((len(manifest_data) + 63) & ~63) - len(manifest_data))
        + payload
    )
    return (
        _PRELUDE.pack(magic, major, minor, codec, len(manifest_data), payload_bytes)
        + body
    )


@pytest.mark.parametrize("codec", [CODEC_NONE, CODEC_ZSTD])
def test_generation_artifact_round_trip_is_deterministic(codec: int) -> None:
    encoded = encode_generation_artifact(_artifact_view(), codec=codec)
    assert (
        encoded.data == encode_generation_artifact(_artifact_view(), codec=codec).data
    )
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


def test_generation_artifact_v1_uncompressed_golden_digest() -> None:
    encoded = encode_generation_artifact(_artifact_view(), codec=CODEC_NONE)
    assert (
        encoded.sha256
        == "f40aa2f833076a63a6a82f2aaeedd99050e6c788b7f51d341ee12af158da933b"
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
            lambda value: value[:10] + struct.pack("<H", 1) + value[12:],
            "minor version",
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
                        routed_experts=np.concatenate(
                            [choice.routed_experts, choice.routed_experts[:1]], axis=0
                        ),
                        router_ids=choice.router_ids,
                        expert_counts=choice.expert_counts,
                    ),
                )
            )
        )

    with pytest.raises(GenerationArtifactFormatError, match="route token count"):
        encode_generation_artifact(
            GenerationArtifactView(
                choices=(
                    GenerationArtifactChoice(
                        choice_index=0,
                        prompt_token_count=2,
                        sequence_token_ids=choice.sequence_token_ids,
                        routed_experts=choice.routed_experts[:-1],
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


def test_encoder_rejects_non_integer_choice_index() -> None:
    choice = _artifact_view().choices[0]
    with pytest.raises(GenerationArtifactFormatError, match="choice_index"):
        encode_generation_artifact(
            GenerationArtifactView(
                choices=(
                    GenerationArtifactChoice(
                        choice_index=1.5,
                        prompt_token_count=choice.prompt_token_count,
                        sequence_token_ids=choice.sequence_token_ids,
                    ),
                )
            )
        )


@pytest.mark.parametrize(
    "sequence_token_ids,routed_experts",
    [
        (np.array([2**63], dtype=np.uint64), None),
        (
            np.array([1], dtype=np.int64),
            np.array([[[2**63]]], dtype=np.uint64),
        ),
    ],
)
def test_encoder_rejects_unsigned_values_outside_wire_range(
    sequence_token_ids: np.ndarray, routed_experts: np.ndarray | None
) -> None:
    with pytest.raises(GenerationArtifactFormatError, match="signed 64-bit"):
        encode_generation_artifact(
            GenerationArtifactView(
                choices=(
                    GenerationArtifactChoice(
                        choice_index=0,
                        prompt_token_count=0,
                        sequence_token_ids=sequence_token_ids,
                        routed_experts=routed_experts,
                        router_ids=(0,) if routed_experts is not None else (),
                        expert_counts=(2**63 + 1,)
                        if routed_experts is not None
                        else (),
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


def test_decoder_rejects_nonzero_padding_and_unknown_component() -> None:
    encoded = bytearray(
        encode_generation_artifact(_artifact_view(), codec=CODEC_NONE).data
    )
    _, _, _, _, manifest_bytes, _ = _PRELUDE.unpack_from(encoded)
    padding_start = _PRELUDE.size + manifest_bytes
    encoded[padding_start] = 1
    with pytest.raises(GenerationArtifactFormatError, match="padding"):
        decode_generation_artifact(bytes(encoded))

    encoded = encode_generation_artifact(_artifact_view(), codec=CODEC_NONE).data
    assert b"moe_routes" in encoded
    with pytest.raises(GenerationArtifactFormatError, match="unsupported component"):
        decode_generation_artifact(encoded.replace(b"moe_routes", b"bad_routes", 1))


def test_decoder_rejects_short_and_invalid_tensor_encoding() -> None:
    with pytest.raises(GenerationArtifactFormatError, match="length"):
        decode_generation_artifact(b"DYNEXP1")

    encoded = encode_generation_artifact(_artifact_view(), codec=CODEC_NONE).data
    assert b"i64" in encoded
    with pytest.raises(GenerationArtifactFormatError, match="tensor dtype"):
        decode_generation_artifact(encoded.replace(b"i64", b"xxx", 1))


def test_decoder_rejects_noncanonical_duplicate_manifest_keys() -> None:
    manifest = b"\x82\xa7choices\x90\xa7choices\x90"
    body = manifest + bytes((64 - len(manifest) % 64) % 64)
    encoded = _PRELUDE.pack(b"DYNEXP1\0", 1, 0, CODEC_NONE, len(manifest), 0) + body
    with pytest.raises(GenerationArtifactFormatError, match="canonical"):
        decode_generation_artifact(encoded)


def test_decoder_rejects_truncated_route_tensor() -> None:
    choice = _artifact_view().choices[0]
    routes = choice.routed_experts[:-1]
    malformed = GenerationArtifactView(
        choices=(
            GenerationArtifactChoice(
                choice_index=choice.choice_index,
                prompt_token_count=choice.prompt_token_count,
                sequence_token_ids=choice.sequence_token_ids,
                routed_experts=routes,
                router_ids=choice.router_ids,
                expert_counts=choice.expert_counts,
            ),
        )
    )
    validated = (
        np.asarray(choice.sequence_token_ids, dtype=np.dtype("<i8")),
        np.asarray(routes, dtype=np.uint8),
        None,
    )
    with patch(
        "dynamo.artifacts.format_v1._validate_choice",
        return_value=validated,
    ):
        encoded = encode_generation_artifact(malformed, codec=CODEC_NONE).data

    with pytest.raises(GenerationArtifactFormatError, match="route token count"):
        decode_generation_artifact(encoded)


def test_decoder_rejects_noncanonical_manifest_key_order() -> None:
    encoded = encode_generation_artifact(_artifact_view(), codec=CODEC_NONE).data
    magic, major, minor, codec, manifest_bytes, payload_bytes = _PRELUDE.unpack_from(
        encoded
    )
    body = encoded[_PRELUDE.size :]
    manifest = msgspec.msgpack.decode(body[:manifest_bytes])
    choice = manifest["choices"][0]
    manifest["choices"][0] = {key: choice[key] for key in reversed(choice)}
    reordered_manifest = msgspec.msgpack.encode(manifest)
    assert len(reordered_manifest) == manifest_bytes
    payload = body[((manifest_bytes + 63) & ~63) :]
    reordered_body = (
        reordered_manifest
        + bytes(((len(reordered_manifest) + 63) & ~63) - len(reordered_manifest))
        + payload
    )
    reordered = (
        _PRELUDE.pack(
            magic, major, minor, codec, len(reordered_manifest), payload_bytes
        )
        + reordered_body
    )
    with pytest.raises(GenerationArtifactFormatError, match="canonical"):
        decode_generation_artifact(reordered)


def test_encoder_rejects_nonfinite_logprobs_and_empty_route_axes() -> None:
    choice = _artifact_view().choices[0]
    with pytest.raises(GenerationArtifactFormatError, match="finite"):
        encode_generation_artifact(
            GenerationArtifactView(
                choices=(
                    GenerationArtifactChoice(
                        choice_index=0,
                        prompt_token_count=2,
                        sequence_token_ids=choice.sequence_token_ids,
                        selected_logprobs=np.array([np.nan, -0.5]),
                        selected_logprobs_token_start=2,
                    ),
                )
            )
        )
    with pytest.raises(GenerationArtifactFormatError, match="routers"):
        encode_generation_artifact(
            GenerationArtifactView(
                choices=(
                    GenerationArtifactChoice(
                        choice_index=0,
                        prompt_token_count=2,
                        sequence_token_ids=choice.sequence_token_ids,
                        routed_experts=np.empty((4, 0, 1), dtype=np.int32),
                    ),
                )
            )
        )


def test_encoder_rejects_logprob_that_overflows_float32() -> None:
    choice = _artifact_view().choices[0]
    overflowing = np.array(
        [float(np.finfo(np.float32).max) * 2, -0.5], dtype=np.float64
    )
    with pytest.raises(GenerationArtifactFormatError, match="finite"):
        encode_generation_artifact(
            GenerationArtifactView(
                choices=(
                    GenerationArtifactChoice(
                        choice_index=0,
                        prompt_token_count=2,
                        sequence_token_ids=choice.sequence_token_ids,
                        selected_logprobs=overflowing,
                        selected_logprobs_token_start=2,
                    ),
                )
            )
        )


def test_decoder_wraps_unhashable_tensor_dtype_as_format_error() -> None:
    encoded = encode_generation_artifact(_artifact_view(), codec=CODEC_NONE).data
    _, _, _, _, manifest_bytes, _ = _PRELUDE.unpack_from(encoded)
    manifest = msgspec.msgpack.decode(
        encoded[_PRELUDE.size : _PRELUDE.size + manifest_bytes]
    )
    manifest["choices"][0]["sequence_token_ids"]["dtype"] = []

    with pytest.raises(GenerationArtifactFormatError, match="tensor dtype"):
        decode_generation_artifact(_replace_manifest(encoded, manifest))


@pytest.mark.parametrize(
    ("sequence_dtype", "route_dtype", "logprob_dtype", "message"),
    [
        (np.dtype("uint8"), np.dtype("uint8"), np.dtype("<f4"), "sequence.*i64"),
        (np.dtype("<i8"), np.dtype("<u2"), np.dtype("<f4"), "route.*dtype"),
        (np.dtype("<i8"), np.dtype("uint8"), np.dtype("<f2"), "logprob.*fp32"),
    ],
)
def test_decoder_rejects_noncanonical_component_dtypes(
    sequence_dtype: np.dtype,
    route_dtype: np.dtype,
    logprob_dtype: np.dtype,
    message: str,
) -> None:
    view = _artifact_view()
    choice = view.choices[0]
    validated = (
        np.asarray(choice.sequence_token_ids, dtype=sequence_dtype),
        np.asarray(choice.routed_experts, dtype=route_dtype),
        np.asarray(choice.selected_logprobs, dtype=logprob_dtype),
    )
    with patch("dynamo.artifacts.format_v1._validate_choice", return_value=validated):
        encoded = encode_generation_artifact(view, codec=CODEC_NONE).data

    with pytest.raises(GenerationArtifactFormatError, match=message):
        decode_generation_artifact(encoded)
