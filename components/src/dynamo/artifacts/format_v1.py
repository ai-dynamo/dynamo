# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict ``generation_artifact_v1`` binary encoder and decoder."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import msgspec
import numpy as np
import zstandard as zstd

MAGIC = b"DYNEXP1\0"
MAJOR_VERSION = 1
MINOR_VERSION = 0
CODEC_NONE = 0
CODEC_ZSTD = 1
_PRELUDE = struct.Struct("<8sHHHIQ")
_ALIGNMENT = 64
_MAX_MANIFEST_BYTES = 1 << 20
_MAX_PAYLOAD_BYTES = 64 << 20
_MAX_OBJECT_BYTES = _PRELUDE.size + _MAX_MANIFEST_BYTES + _MAX_PAYLOAD_BYTES + 65536

_NUMPY_TO_WIRE = {
    np.dtype("uint8"): "u8",
    np.dtype("<u2"): "u16",
    np.dtype("<i4"): "i32",
    np.dtype("<i8"): "i64",
    np.dtype("<f2"): "fp16",
    np.dtype("<f4"): "fp32",
}
_WIRE_TO_NUMPY = {wire: dtype for dtype, wire in _NUMPY_TO_WIRE.items()}


class GenerationArtifactFormatError(ValueError):
    """Raised when an artifact violates the version 1 wire contract."""


def _immutable_array(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(value).copy()
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class GenerationArtifactChoice:
    choice_index: int
    prompt_token_count: int
    sequence_token_ids: np.ndarray
    routed_experts: np.ndarray | None = None
    router_ids: tuple[int, ...] = ()
    expert_counts: tuple[int, ...] = ()
    selected_logprobs: np.ndarray | None = None
    selected_logprobs_token_start: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sequence_token_ids", _immutable_array(self.sequence_token_ids)
        )
        if self.routed_experts is not None:
            object.__setattr__(
                self, "routed_experts", _immutable_array(self.routed_experts)
            )
        if self.selected_logprobs is not None:
            object.__setattr__(
                self, "selected_logprobs", _immutable_array(self.selected_logprobs)
            )
        object.__setattr__(self, "router_ids", tuple(self.router_ids))
        object.__setattr__(self, "expert_counts", tuple(self.expert_counts))


@dataclass(frozen=True)
class GenerationArtifactView:
    choices: tuple[GenerationArtifactChoice, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "choices", tuple(self.choices))


@dataclass(frozen=True)
class EncodedGenerationArtifact:
    data: bytes
    byte_count: int
    sha256: str


@dataclass(frozen=True)
class DecodedGenerationArtifact:
    manifest: Mapping[str, Any]
    choices: tuple[GenerationArtifactChoice, ...]


def _align_up(value: int) -> int:
    return (value + _ALIGNMENT - 1) & ~(_ALIGNMENT - 1)


def _integer_array(value: Any, field: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.dtype == np.dtype("bool"):
        raise GenerationArtifactFormatError(f"{field} must use an integer dtype")
    if array.size and int(array.min()) < 0:
        raise GenerationArtifactFormatError(f"{field} must not contain negative values")
    maximum = int(array.max()) if array.size else 0
    if maximum <= np.iinfo(np.uint8).max:
        dtype = np.dtype("uint8")
    elif maximum <= np.iinfo(np.uint16).max:
        dtype = np.dtype("<u2")
    elif maximum <= np.iinfo(np.int32).max:
        dtype = np.dtype("<i4")
    else:
        dtype = np.dtype("<i8")
    return np.ascontiguousarray(array, dtype=dtype)


def _token_array(value: Any) -> np.ndarray:
    array = _integer_array(value, "sequence_token_ids")
    if array.ndim != 1:
        raise GenerationArtifactFormatError("sequence_token_ids must have rank 1")
    return np.ascontiguousarray(array, dtype=np.dtype("<i8"))


def _logprob_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind != "f" or array.ndim != 1:
        raise GenerationArtifactFormatError(
            "selected logprobs must use a rank 1 floating dtype"
        )
    if not np.isfinite(array).all():
        raise GenerationArtifactFormatError("selected logprobs must be finite")
    return np.ascontiguousarray(array, dtype=np.dtype("<f4"))


def _tensor_ref(array: np.ndarray, offset: int) -> dict[str, Any]:
    dtype = _NUMPY_TO_WIRE.get(array.dtype)
    if dtype is None:
        raise GenerationArtifactFormatError(f"unsupported tensor dtype {array.dtype}")
    return {
        "dtype": dtype,
        "shape": list(array.shape),
        "offset": offset,
        "byte_count": array.nbytes,
    }


def _validate_choice(
    choice: GenerationArtifactChoice,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    sequence = _token_array(choice.sequence_token_ids)
    sequence_length = len(sequence)
    if isinstance(choice.choice_index, bool) or choice.choice_index < 0:
        raise GenerationArtifactFormatError("choice_index must be non-negative")
    if isinstance(choice.prompt_token_count, bool) or not isinstance(
        choice.prompt_token_count, int
    ):
        raise GenerationArtifactFormatError("prompt_token_count must be an integer")
    if not 0 <= choice.prompt_token_count <= sequence_length:
        raise GenerationArtifactFormatError(
            "prompt_token_count exceeds sequence length"
        )

    routes = None
    if choice.routed_experts is not None:
        routes = _integer_array(choice.routed_experts, "moe_routes.expert_ids")
        if routes.ndim != 3:
            raise GenerationArtifactFormatError("moe routes must have rank 3")
        if routes.shape[1] == 0 or routes.shape[2] == 0:
            raise GenerationArtifactFormatError(
                "moe routes must contain routers and selected experts"
            )
        if routes.shape[0] > sequence_length:
            raise GenerationArtifactFormatError(
                "route token count exceeds the sequence token count"
            )
        routers = routes.shape[1]
        if len(choice.router_ids) != routers or len(choice.expert_counts) != routers:
            raise GenerationArtifactFormatError(
                "router_ids and expert_counts must match the route router dimension"
            )
        if any(
            isinstance(router_id, bool)
            or not isinstance(router_id, int)
            or router_id < 0
            for router_id in choice.router_ids
        ):
            raise GenerationArtifactFormatError(
                "router_ids must be non-negative integers"
            )
        if len(set(choice.router_ids)) != routers:
            raise GenerationArtifactFormatError("router_ids must be unique")
        if any(
            isinstance(count, bool) or not isinstance(count, int) or count <= 0
            for count in choice.expert_counts
        ):
            raise GenerationArtifactFormatError("expert_counts must be positive")
        for router_index, expert_count in enumerate(choice.expert_counts):
            values = routes[:, router_index, :]
            if values.size and int(values.max()) >= expert_count:
                raise GenerationArtifactFormatError(
                    f"expert ID exceeds expert count for router {router_index}"
                )

    selected = None
    if choice.selected_logprobs is not None:
        selected = _logprob_array(choice.selected_logprobs)
        start = choice.selected_logprobs_token_start
        if isinstance(start, bool) or not isinstance(start, int):
            raise GenerationArtifactFormatError(
                "selected logprob token_start must be an integer"
            )
        if start < 0 or start + len(selected) > sequence_length:
            raise GenerationArtifactFormatError(
                "selected logprob rows exceed the sequence token range"
            )
        if start != choice.prompt_token_count or len(selected) != (
            sequence_length - choice.prompt_token_count
        ):
            raise GenerationArtifactFormatError(
                "selected logprobs must align with every completion token"
            )
    return sequence, routes, selected


def encode_generation_artifact(
    view: GenerationArtifactView, *, codec: int = CODEC_ZSTD
) -> EncodedGenerationArtifact:
    """Encode a deterministic, bounded version 1 generation artifact."""
    if codec not in (CODEC_NONE, CODEC_ZSTD):
        raise GenerationArtifactFormatError(f"unsupported codec {codec}")
    if not view.choices:
        raise GenerationArtifactFormatError("at least one choice is required")
    indexes = [choice.choice_index for choice in view.choices]
    if indexes != sorted(set(indexes)):
        raise GenerationArtifactFormatError("choice indexes must be unique and ordered")

    manifest_choices: list[dict[str, Any]] = []
    payload_parts: list[bytes] = []
    payload_offset = 0
    for choice in view.choices:
        sequence, routes, selected = _validate_choice(choice)
        sequence_ref = _tensor_ref(sequence, payload_offset)
        payload_parts.append(sequence.tobytes(order="C"))
        payload_offset += sequence.nbytes
        components: list[dict[str, Any]] = []

        if routes is not None:
            route_ref = _tensor_ref(routes, payload_offset)
            payload_parts.append(routes.tobytes(order="C"))
            payload_offset += routes.nbytes
            components.append(
                {
                    "kind": "moe_routes",
                    "token_start": 0,
                    "expert_ids": route_ref,
                    "router_ids": list(choice.router_ids),
                    "expert_counts": list(choice.expert_counts),
                }
            )
        if selected is not None:
            selected_ref = _tensor_ref(selected, payload_offset)
            payload_parts.append(selected.tobytes(order="C"))
            payload_offset += selected.nbytes
            components.append(
                {
                    "kind": "selected_logprobs",
                    "token_start": choice.selected_logprobs_token_start,
                    "logprobs": selected_ref,
                }
            )
        manifest_choices.append(
            {
                "choice_index": choice.choice_index,
                "prompt_token_count": choice.prompt_token_count,
                "sequence_token_ids": sequence_ref,
                "components": components,
            }
        )

    payload = b"".join(payload_parts)
    if len(payload) > _MAX_PAYLOAD_BYTES:
        raise GenerationArtifactFormatError("artifact payload exceeds configured limit")
    manifest = msgspec.msgpack.encode({"choices": manifest_choices})
    if len(manifest) > _MAX_MANIFEST_BYTES:
        raise GenerationArtifactFormatError(
            "artifact manifest exceeds configured limit"
        )
    body = manifest + bytes(_align_up(len(manifest)) - len(manifest)) + payload
    encoded_body = (
        body
        if codec == CODEC_NONE
        else zstd.ZstdCompressor(level=1, write_content_size=True).compress(body)
    )
    prelude = _PRELUDE.pack(
        MAGIC,
        MAJOR_VERSION,
        MINOR_VERSION,
        codec,
        len(manifest),
        len(payload),
    )
    data = prelude + encoded_body
    return EncodedGenerationArtifact(
        data=data,
        byte_count=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _strict_keys(value: Mapping[str, Any], expected: set[str], field: str) -> None:
    if set(value) != expected:
        raise GenerationArtifactFormatError(f"invalid {field} fields")


def _read_tensor(
    reference: Mapping[str, Any], payload: memoryview, expected_offset: int
) -> tuple[np.ndarray, int]:
    if not isinstance(reference, Mapping):
        raise GenerationArtifactFormatError("tensor reference must be an object")
    _strict_keys(reference, {"dtype", "shape", "offset", "byte_count"}, "tensor")
    dtype = _WIRE_TO_NUMPY.get(reference["dtype"])
    if dtype is None:
        raise GenerationArtifactFormatError("unsupported tensor dtype")
    shape = reference["shape"]
    if not isinstance(shape, list) or any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
    ):
        raise GenerationArtifactFormatError("invalid tensor shape")
    offset = reference["offset"]
    byte_count = reference["byte_count"]
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count < 0
    ):
        raise GenerationArtifactFormatError("invalid tensor byte range")
    if offset != expected_offset:
        raise GenerationArtifactFormatError("tensor ranges must be ordered and gapless")
    elements = 1
    for dim in shape:
        elements *= dim
        if elements > _MAX_PAYLOAD_BYTES:
            raise GenerationArtifactFormatError("tensor shape exceeds limit")
    expected_bytes = elements * dtype.itemsize
    if byte_count != expected_bytes or offset + byte_count > len(payload):
        raise GenerationArtifactFormatError("invalid tensor byte range")
    array = np.frombuffer(payload[offset : offset + byte_count], dtype=dtype).reshape(
        shape
    )
    array.setflags(write=False)
    return array, offset + byte_count


def decode_generation_artifact(data: bytes) -> DecodedGenerationArtifact:
    """Decode and validate one complete version 1 artifact."""
    if len(data) < _PRELUDE.size or len(data) > _MAX_OBJECT_BYTES:
        raise GenerationArtifactFormatError("invalid artifact length")
    magic, major, minor, codec, manifest_bytes, payload_bytes = _PRELUDE.unpack_from(
        data
    )
    if magic != MAGIC:
        raise GenerationArtifactFormatError("invalid artifact magic")
    if major != MAJOR_VERSION or minor != MINOR_VERSION:
        raise GenerationArtifactFormatError("unsupported major version")
    if codec not in (CODEC_NONE, CODEC_ZSTD):
        raise GenerationArtifactFormatError("unsupported codec")
    if manifest_bytes > _MAX_MANIFEST_BYTES or payload_bytes > _MAX_PAYLOAD_BYTES:
        raise GenerationArtifactFormatError("declared length exceeds configured limit")
    decoded_size = _align_up(manifest_bytes) + payload_bytes
    encoded_body = data[_PRELUDE.size :]
    if codec == CODEC_NONE:
        if len(encoded_body) != decoded_size:
            raise GenerationArtifactFormatError("invalid decoded body length")
        body = encoded_body
    else:
        try:
            params = zstd.get_frame_parameters(encoded_body)
            if params.dict_id != 0 or params.content_size != decoded_size:
                raise GenerationArtifactFormatError("invalid zstd frame parameters")
            body = zstd.ZstdDecompressor().decompress(
                encoded_body,
                max_output_size=decoded_size,
                allow_extra_data=False,
            )
        except zstd.ZstdError as exc:
            raise GenerationArtifactFormatError("invalid zstd body") from exc
        if len(body) != decoded_size:
            raise GenerationArtifactFormatError("invalid decoded body length")

    padding_start = manifest_bytes
    payload_start = _align_up(manifest_bytes)
    if any(body[padding_start:payload_start]):
        raise GenerationArtifactFormatError("alignment padding must be zero")
    try:
        manifest_data = body[:manifest_bytes]
        manifest = msgspec.msgpack.decode(manifest_data)
    except msgspec.DecodeError as exc:
        raise GenerationArtifactFormatError("invalid MessagePack manifest") from exc
    if not isinstance(manifest, dict):
        raise GenerationArtifactFormatError("manifest must be an object")
    if msgspec.msgpack.encode(manifest) != manifest_data:
        raise GenerationArtifactFormatError("manifest encoding is not canonical")
    _strict_keys(manifest, {"choices"}, "manifest")
    raw_choices = manifest["choices"]
    if not isinstance(raw_choices, list) or not raw_choices:
        raise GenerationArtifactFormatError("manifest choices must be non-empty")

    payload = memoryview(body)[payload_start:]
    offset = 0
    choices: list[GenerationArtifactChoice] = []
    previous_index = -1
    for raw_choice in raw_choices:
        if not isinstance(raw_choice, dict):
            raise GenerationArtifactFormatError("choice must be an object")
        _strict_keys(
            raw_choice,
            {"choice_index", "prompt_token_count", "sequence_token_ids", "components"},
            "choice",
        )
        choice_index = raw_choice["choice_index"]
        if (
            isinstance(choice_index, bool)
            or not isinstance(choice_index, int)
            or choice_index < 0
            or choice_index <= previous_index
        ):
            raise GenerationArtifactFormatError(
                "choice indexes must be unique and ordered"
            )
        previous_index = choice_index
        prompt_token_count = raw_choice["prompt_token_count"]
        if isinstance(prompt_token_count, bool) or not isinstance(
            prompt_token_count, int
        ):
            raise GenerationArtifactFormatError("prompt_token_count must be an integer")
        sequence, offset = _read_tensor(
            raw_choice["sequence_token_ids"], payload, offset
        )
        routes = None
        router_ids: tuple[int, ...] = ()
        expert_counts: tuple[int, ...] = ()
        selected = None
        selected_start = 0
        seen: set[str] = set()
        components = raw_choice["components"]
        if not isinstance(components, list):
            raise GenerationArtifactFormatError("components must be an array")
        for component in components:
            if not isinstance(component, dict) or not isinstance(
                component.get("kind"), str
            ):
                raise GenerationArtifactFormatError("invalid component")
            kind = component["kind"]
            if kind in seen:
                raise GenerationArtifactFormatError("duplicate component kind")
            seen.add(kind)
            if kind == "moe_routes":
                _strict_keys(
                    component,
                    {
                        "kind",
                        "token_start",
                        "expert_ids",
                        "router_ids",
                        "expert_counts",
                    },
                    "moe_routes",
                )
                if (
                    isinstance(component["token_start"], bool)
                    or not isinstance(component["token_start"], int)
                    or component["token_start"] != 0
                ):
                    raise GenerationArtifactFormatError(
                        "moe route token_start must be zero"
                    )
                routes, offset = _read_tensor(component["expert_ids"], payload, offset)
                raw_router_ids = component["router_ids"]
                raw_expert_counts = component["expert_counts"]
                if (
                    not isinstance(raw_router_ids, list)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value < 0
                        for value in raw_router_ids
                    )
                    or not isinstance(raw_expert_counts, list)
                    or any(
                        isinstance(value, bool)
                        or not isinstance(value, int)
                        or value <= 0
                        for value in raw_expert_counts
                    )
                ):
                    raise GenerationArtifactFormatError("invalid router metadata")
                router_ids = tuple(raw_router_ids)
                expert_counts = tuple(raw_expert_counts)
            elif kind == "selected_logprobs":
                _strict_keys(component, {"kind", "token_start", "logprobs"}, kind)
                selected_start = component["token_start"]
                if isinstance(selected_start, bool) or not isinstance(
                    selected_start, int
                ):
                    raise GenerationArtifactFormatError(
                        "selected logprob token_start must be an integer"
                    )
                selected, offset = _read_tensor(component["logprobs"], payload, offset)
            else:
                raise GenerationArtifactFormatError("unsupported component kind")
        choice = GenerationArtifactChoice(
            choice_index=choice_index,
            prompt_token_count=prompt_token_count,
            sequence_token_ids=sequence,
            routed_experts=routes,
            router_ids=router_ids,
            expert_counts=expert_counts,
            selected_logprobs=selected,
            selected_logprobs_token_start=selected_start,
        )
        _validate_choice(choice)
        choices.append(choice)
    if offset != payload_bytes:
        raise GenerationArtifactFormatError("tensor ranges do not cover payload length")
    return DecodedGenerationArtifact(manifest=manifest, choices=tuple(choices))
