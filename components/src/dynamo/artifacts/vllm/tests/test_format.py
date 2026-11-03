# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden geometry and malformed-object tests for the public v1 container."""

import hashlib
import struct
from dataclasses import replace

import pytest

msgspec = pytest.importorskip("msgspec")
np = pytest.importorskip("numpy")
zstd = pytest.importorskip("zstandard")

from dynamo.artifacts.vllm.format import (  # noqa: E402
    CODEC_NONE,
    CODEC_ZSTD,
    GenerationArtifactChoice,
    GenerationArtifactFormatError,
    GenerationArtifactView,
    MoeRoutes,
    RouterRoutes,
    SelectedLogprobs,
    Tensor,
    TopKLogprobs,
    compact_routes,
    decode_generation_artifact,
    encode_generation_artifact,
    iter_generation_artifact,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
]
PRELUDE = struct.Struct("<8sHHHIQ")


def _tensor(values, dtype="i64"):
    return Tensor.from_array(np.asarray(values), dtype=dtype)


def _view(*components, tokens=(11, 12, 13, 14), prompt=2):
    return GenerationArtifactView(
        (GenerationArtifactChoice(0, prompt, _tensor(tokens), components),)
    )


def _unpack(data):
    fields = PRELUDE.unpack_from(data)
    body = data[PRELUDE.size :]
    if fields[3] == CODEC_ZSTD:
        body = zstd.ZstdDecompressor().decompress(body)
    manifest_length = fields[4]
    return (
        msgspec.msgpack.decode(body[:manifest_length]),
        body[(manifest_length + 63) // 64 * 64 :],
    )


def _wire(manifest, payload):
    encoded = msgspec.msgpack.encode(manifest, order="deterministic")
    return (
        PRELUDE.pack(b"DYNEXP1\0", 1, 0, CODEC_NONE, len(encoded), len(payload))
        + encoded
        + bytes(-len(encoded) % 64)
        + payload
    )


@pytest.mark.parametrize("codec", [CODEC_NONE, CODEC_ZSTD])
def test_token_only_roundtrip_and_whole_object_digest(codec):
    result = encode_generation_artifact(_view(), codec=codec)
    decoded = decode_generation_artifact(result.data)
    assert decoded == _view()
    assert result.byte_count == len(result.data)
    assert result.sha256 == hashlib.sha256(result.data).hexdigest()
    assert result.data == b"".join(iter_generation_artifact(_view(), codec=codec))


@pytest.mark.parametrize("codec", [CODEC_NONE, CODEC_ZSTD])
@pytest.mark.parametrize("as_view", [False, True])
def test_decode_accepts_mutable_buffers_without_retaining_them(codec, as_view):
    encoded = encode_generation_artifact(_view(), codec=codec)
    source = bytearray(encoded.data)
    data = memoryview(source) if as_view else source
    decoded = decode_generation_artifact(data)
    source[-1] ^= 1
    assert decoded == _view()


def test_decode_rejects_noncontiguous_buffer():
    encoded = encode_generation_artifact(_view()).data
    with pytest.raises(GenerationArtifactFormatError, match="buffer"):
        decode_generation_artifact(memoryview(encoded)[::2])


def test_unknown_minor_version_requires_explicit_reader_support():
    encoded = encode_generation_artifact(_view()).data
    magic, major, _, codec, manifest_bytes, payload_bytes = PRELUDE.unpack_from(encoded)
    header = PRELUDE.pack(magic, major, 1, codec, manifest_bytes, payload_bytes)
    with pytest.raises(GenerationArtifactFormatError, match="version"):
        decode_generation_artifact(header + encoded[PRELUDE.size :])


@pytest.mark.parametrize("experts,expected", [(256, "u8"), (257, "u16")])
def test_dtype_comes_from_geometry_not_observed_ids(experts, expected):
    routes = MoeRoutes(1, (RouterRoutes(7, experts, _tensor([[1, 2], [2, 1]])),))
    result = encode_generation_artifact(_view(routes))
    manifest, _ = _unpack(result.data)
    component = manifest["choices"][0]["components"][0]
    assert set(component) == {"kind", "token_start", "routers"}
    assert component["routers"][0]["expert_ids"]["dtype"] == expected
    decoded = decode_generation_artifact(result.data)
    assert decoded.choices[0].components[0].routers[0].expert_ids.dtype == expected


def test_heterogeneous_router_widths_and_immutable_input():
    first = RouterRoutes(3, 256, _tensor([[4], [5]]))
    second = RouterRoutes(9, 257, _tensor([[1, 256], [256, 1]]))
    result = encode_generation_artifact(_view(MoeRoutes(1, (first, second))))
    decoded = decode_generation_artifact(result.data).choices[0].components[0]
    assert tuple(router.expert_ids.shape for router in decoded.routers) == (
        (2, 1),
        (2, 2),
    )
    assert tuple(router.expert_ids.dtype for router in decoded.routers) == ("u8", "u16")


@pytest.mark.parametrize(
    "prompt,start,rows", [(2, 0, 3), (2, 1, 2), (2, 3, 0), (4, 0, 4), (4, 4, 0)]
)
def test_route_suffix_and_prefill_extent(prompt, start, rows):
    ids = Tensor.from_array(np.ones((rows, 1), dtype=np.int32))
    routes = MoeRoutes(start, (RouterRoutes(0, 2, ids),))
    decoded = decode_generation_artifact(
        encode_generation_artifact(_view(routes, prompt=prompt)).data
    )
    assert decoded.choices[0].components[0].routers[0].expert_ids.shape == (rows, 1)


@pytest.mark.parametrize(
    "ids,experts",
    [([[1, 1], [0, 1]], 2), ([[0, 2], [0, 1]], 2), ([[0, 1, 2], [0, 1, 2]], 2)],
)
def test_invalid_expert_rows_rejected(ids, experts):
    routes = MoeRoutes(1, (RouterRoutes(0, experts, _tensor(ids)),))
    with pytest.raises(GenerationArtifactFormatError):
        encode_generation_artifact(_view(routes))


def test_compact_native_routes_preserves_router_order_and_owns_bytes():
    native = np.asarray([[[1, 0], [0, 256]], [[0, 1], [256, 0]]], dtype=np.int64)
    routes = compact_routes(native, (5, 2), (256, 257), token_start=1)
    native.fill(0)
    assert tuple(router.router_id for router in routes.routers) == (5, 2)
    assert routes.routers[1].expert_ids.as_array().tolist() == [[0, 256], [256, 0]]
    assert not routes.routers[0].expert_ids.as_array().flags.writeable


def test_selected_prompt_logprobs_and_topk_pairing():
    selected = SelectedLogprobs(1, _tensor([-0.1, -0.2], "fp16"))
    topk = TopKLogprobs(
        2,
        _tensor([[13, 20], [14, 21]], "i32"),
        _tensor([[-0.2, -1.0], [-0.3, -2.0]], "fp32"),
    )
    decoded = decode_generation_artifact(
        encode_generation_artifact(_view(selected, topk)).data
    )
    assert decoded == _view(selected, topk)


def test_bfloat16_preserves_raw_bits():
    selected = SelectedLogprobs(
        2, Tensor("bf16", (2,), struct.pack("<HH", 0xBF80, 0xC000))
    )
    decoded = decode_generation_artifact(
        encode_generation_artifact(_view(selected)).data
    )
    assert decoded.choices[0].components[0].logprobs == selected.logprobs


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("value", ["nan", "negative_nan", "positive_infinity"])
def test_nonfinite_logprobs_rejected_on_encode_and_decode(dtype, value):
    patterns = {
        "fp16": {"nan": 0x7E00, "negative_nan": 0xFE00, "positive_infinity": 0x7C00},
        "bf16": {"nan": 0x7FC0, "negative_nan": 0xFFC0, "positive_infinity": 0x7F80},
        "fp32": {
            "nan": 0x7FC00000,
            "negative_nan": 0xFFC00000,
            "positive_infinity": 0x7F800000,
        },
    }
    bits = patterns[dtype][value]
    raw = struct.pack("<I" if dtype == "fp32" else "<H", bits)
    bad = Tensor(dtype, (1,), raw)
    with pytest.raises(GenerationArtifactFormatError, match="logprob"):
        encode_generation_artifact(_view(SelectedLogprobs(2, bad)))
    with pytest.raises(GenerationArtifactFormatError, match="logprob"):
        encode_generation_artifact(
            _view(TopKLogprobs(2, _tensor([[13]], "i32"), Tensor(dtype, (1, 1), raw)))
        )

    valid = encode_generation_artifact(
        _view(SelectedLogprobs(2, Tensor(dtype, (1,), bytes(len(raw))))),
        codec=CODEC_NONE,
    )
    manifest, payload = _unpack(valid.data)
    with pytest.raises(GenerationArtifactFormatError, match="logprob"):
        decode_generation_artifact(_wire(manifest, payload[: -len(raw)] + raw))
    valid_topk = encode_generation_artifact(
        _view(
            TopKLogprobs(
                2, _tensor([[13]], "i32"), Tensor(dtype, (1, 1), bytes(len(raw)))
            )
        ),
        codec=CODEC_NONE,
    )
    manifest, payload = _unpack(valid_topk.data)
    with pytest.raises(GenerationArtifactFormatError, match="logprob"):
        decode_generation_artifact(_wire(manifest, payload[: -len(raw)] + raw))


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
def test_negative_infinity_logprob_is_supported(dtype):
    raw = struct.pack(
        "<I" if dtype == "fp32" else "<H",
        {"fp16": 0xFC00, "bf16": 0xFF80, "fp32": 0xFF800000}[dtype],
    )
    selected = SelectedLogprobs(2, Tensor(dtype, (1,), raw))
    assert (
        decode_generation_artifact(encode_generation_artifact(_view(selected)).data)
        .choices[0]
        .components[0]
        == selected
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_float_tensor_rejects_nonfinite_source_except_negative_infinity(value):
    with pytest.raises(GenerationArtifactFormatError, match="floating"):
        Tensor.from_array(np.asarray([value]), dtype="fp16")
    assert np.isneginf(
        Tensor.from_array(np.asarray([float("-inf")]), dtype="fp16").as_array()[0]
    )


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "overlap", "gap", "shape", "duplicate", "route_dtype", "route_count"],
)
def test_malformed_manifest_rejected(mutation):
    routes = MoeRoutes(1, (RouterRoutes(0, 257, _tensor([[0], [1]])),))
    manifest, payload = _unpack(encode_generation_artifact(_view(routes)).data)
    choice = manifest["choices"][0]
    router = choice["components"][0]["routers"][0]
    if mutation == "unknown":
        choice["unexpected"] = 1
    elif mutation == "overlap":
        router["expert_ids"]["offset"] = 0
    elif mutation == "gap":
        router["expert_ids"]["offset"] += 1
    elif mutation == "shape":
        router["expert_ids"]["shape"] = [2**63, 2**63]
    elif mutation == "duplicate":
        choice["components"].append(choice["components"][0])
    elif mutation == "route_dtype":
        router["expert_count"] = 256
    else:
        router["expert_count"] = True
    with pytest.raises(GenerationArtifactFormatError):
        decode_generation_artifact(_wire(manifest, payload))


@pytest.mark.parametrize(
    "mutation",
    [
        "magic",
        "version",
        "codec",
        "oversize",
        "trailing",
        "truncated",
        "padding",
        "frame",
    ],
)
def test_invalid_container_rejected(mutation):
    result = encode_generation_artifact(_view(), codec=CODEC_NONE)
    data = result.data
    fields = list(PRELUDE.unpack_from(data))
    if mutation == "magic":
        fields[0] = b"NOTEXP1\0"
    elif mutation == "version":
        fields[1] = 2
    elif mutation == "codec":
        fields[3] = 77
    elif mutation == "oversize":
        fields[5] = 1 << 40
    elif mutation == "trailing":
        data += b"extra"
    elif mutation == "truncated":
        data = data[:-1]
    elif mutation == "padding":
        offset = PRELUDE.size + fields[4]
        data = data[:offset] + b"x" + data[offset + 1 :]
    else:
        fields[3] = CODEC_ZSTD
        data = (
            data[: PRELUDE.size]
            + zstd.ZstdCompressor().compress(data[PRELUDE.size :])
            + zstd.ZstdCompressor().compress(b"extra")
        )
    data = PRELUDE.pack(*fields) + data[PRELUDE.size :]
    with pytest.raises(GenerationArtifactFormatError):
        decode_generation_artifact(data)


def test_duplicate_messagepack_keys_rejected():
    manifest, payload = _unpack(encode_generation_artifact(_view()).data)
    pair = msgspec.msgpack.encode("choices") + msgspec.msgpack.encode(
        manifest["choices"], order="deterministic"
    )
    duplicate = b"\x82" + pair + pair
    data = (
        PRELUDE.pack(b"DYNEXP1\0", 1, 0, 0, len(duplicate), len(payload))
        + duplicate
        + bytes(-len(duplicate) % 64)
        + payload
    )
    with pytest.raises(GenerationArtifactFormatError):
        decode_generation_artifact(data)


def test_multiple_choices_and_order_rejection():
    first = _view().choices[0]
    second = replace(first, choice_index=2)
    view = GenerationArtifactView((first, second))
    assert decode_generation_artifact(encode_generation_artifact(view).data) == view
    with pytest.raises(GenerationArtifactFormatError):
        encode_generation_artifact(GenerationArtifactView((second, first)))


def test_reader_binds_full_router_profile_to_model_capability():
    routes = compact_routes(
        np.zeros((2, 2, 1), dtype=np.int32), (9, 3), (4, 4), token_start=1
    )
    encoded = encode_generation_artifact(_view(routes))
    decode_generation_artifact(encoded.data, expected_router_profile=((9, 4), (3, 4)))
    with pytest.raises(GenerationArtifactFormatError, match="router profile"):
        decode_generation_artifact(
            encoded.data, expected_router_profile=((3, 4), (9, 4))
        )
    with pytest.raises(GenerationArtifactFormatError, match="router profile"):
        decode_generation_artifact(
            encoded.data, expected_router_profile=((9, 4), (3, 5))
        )

    inflated = np.zeros((2, 2, 1), dtype=np.int32)
    inflated[:, 1, :] = 4
    forged = compact_routes(inflated, (9, 3), (4, 5), token_start=1)
    forged_data = encode_generation_artifact(_view(forged)).data
    with pytest.raises(GenerationArtifactFormatError, match="router profile"):
        decode_generation_artifact(
            forged_data, expected_router_profile=((9, 4), (3, 4))
        )


@pytest.mark.parametrize("dtype", ["u8", "u16", "i32", "i64"])
def test_sequence_integer_representations(dtype):
    choice = GenerationArtifactChoice(0, 1, _tensor([1, 2, 3], dtype))
    view = GenerationArtifactView((choice,))
    assert decode_generation_artifact(encode_generation_artifact(view).data) == view


@pytest.mark.parametrize(
    "field,value",
    [
        ("dtype", "unknown"),
        ("shape", (1 << 40,)),
        ("shape", (True,)),
        ("data", bytearray(8)),
    ],
)
def test_tensor_validates_before_copy(field, value):
    fields = {"dtype": "i64", "shape": (1,), "data": bytes(8), field: value}
    with pytest.raises(GenerationArtifactFormatError):
        Tensor(**fields)


def test_short_invalid_zstd_and_noncanonical_manifest():
    header = PRELUDE.pack(b"DYNEXP1\0", 1, 0, CODEC_ZSTD, 1, 0)
    with pytest.raises(GenerationArtifactFormatError, match="zstd"):
        decode_generation_artifact(header + b"broken")
    header = PRELUDE.pack(b"DYNEXP1\0", 1, 0, CODEC_NONE, 1, 0)
    with pytest.raises(GenerationArtifactFormatError, match="MessagePack"):
        decode_generation_artifact(header + b"\xc1" + bytes(63))


@pytest.mark.parametrize(
    "component",
    [
        SelectedLogprobs(4, _tensor([0.1], "fp32")),
        SelectedLogprobs(0, _tensor([1], "i32")),
        TopKLogprobs(0, _tensor([[1, 2]]), _tensor([[0.1]], "fp32")),
        MoeRoutes(1, (RouterRoutes(1, 65537, _tensor([[1], [2]])),)),
        MoeRoutes(4, (RouterRoutes(1, 4, _tensor([[1], [2]])),)),
    ],
)
def test_component_semantics_rejected(component):
    with pytest.raises(GenerationArtifactFormatError):
        encode_generation_artifact(_view(component))


def test_aggregate_bound_checked_before_route_conversion(monkeypatch):
    routes = MoeRoutes(1, (RouterRoutes(0, 4, _tensor([[1], [2]])),))
    monkeypatch.setattr("dynamo.artifacts.vllm.format.MAX_PAYLOAD_BYTES", 40)
    with pytest.raises(GenerationArtifactFormatError, match="capture payload"):
        encode_generation_artifact(_view(routes))


def test_tensor_owns_source_and_checks_narrowing():
    source = np.asarray([1, 2], dtype=np.int64)
    tensor = Tensor.from_array(source)
    source.fill(7)
    assert tensor.as_array().tolist() == [1, 2]
    with pytest.raises(GenerationArtifactFormatError, match="out of range"):
        Tensor.from_array(np.asarray([256]), dtype="u8")


@pytest.mark.parametrize(
    "source", [np.asarray([1.0], dtype=np.float64), np.asarray([True])]
)
def test_unsupported_inferred_source_dtype_has_specific_error(source):
    with pytest.raises(GenerationArtifactFormatError, match="unsupported source dtype"):
        Tensor.from_array(source)


def test_floating_conversion_checks_range_even_with_negative_infinity():
    source = np.asarray([float("-inf"), -70000.0], dtype=np.float32)
    with pytest.raises(GenerationArtifactFormatError, match="out of range"):
        Tensor.from_array(source, dtype="fp16")


def test_floating_conversion_rejects_nan_and_positive_infinity():
    for value in (float("nan"), float("inf")):
        with pytest.raises(GenerationArtifactFormatError, match="nonfinite"):
            Tensor.from_array(np.asarray([value], dtype=np.float32), dtype="fp16")
