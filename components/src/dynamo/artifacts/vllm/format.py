# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded producer and strict reader for generation_artifact_v1 (wire version 1.0).

Logprob tensors allow finite values and -Inf (zero probability), but not NaN or +Inf.
Only minor version 0 is defined here. A later additive minor revision needs an
explicit schema and capability update before this reader accepts it.
"""

from __future__ import annotations

import hashlib
import io
import struct
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

import msgspec
import numpy as np
import zstandard as zstd

CODEC_NONE = 0
CODEC_ZSTD = 1
MAX_PAYLOAD_BYTES = 64 << 20
MAX_MANIFEST_BYTES = 1 << 20
MAX_OBJECT_BYTES = MAX_PAYLOAD_BYTES + MAX_MANIFEST_BYTES + (1 << 20)
_PRELUDE = struct.Struct("<8sHHHIQ")
_MAX_CHOICES = 32
_MAX_ROUTERS = 4096
_MAX_DIMENSION = 1 << 20
_MAX_TOPK = 1024
_MAX_TENSORS = 4096
_DTYPES: dict[str, np.dtype[Any]] = {
    "u8": np.dtype("u1"),
    "u16": np.dtype("<u2"),
    "i32": np.dtype("<i4"),
    "i64": np.dtype("<i8"),
    "fp16": np.dtype("<f2"),
    "bf16": np.dtype("<u2"),
    "fp32": np.dtype("<f4"),
}
_INTEGER_DTYPES = frozenset({"u8", "u16", "i32", "i64"})
_FLOAT_DTYPES = frozenset({"fp16", "bf16", "fp32"})


class GenerationArtifactFormatError(ValueError):
    """The object or capture does not satisfy the v1 contract and local limits."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GenerationArtifactFormatError(message)


def _integer(value: Any, low: int, high: int, field: str) -> int:
    _require(type(value) is int and low <= value <= high, f"invalid {field}")
    return value


def _tensor_size(dtype: str, shape: tuple[int, ...]) -> int:
    _require(isinstance(dtype, str) and dtype in _DTYPES, "unsupported tensor dtype")
    _require(isinstance(shape, tuple) and 1 <= len(shape) <= 2, "invalid tensor rank")
    elements = 1
    for dimension in shape:
        _integer(dimension, 0, _MAX_DIMENSION, "dimension")
        elements *= dimension
        _require(elements <= MAX_PAYLOAD_BYTES, "tensor shape exceeds limit")
    byte_count = elements * _DTYPES[dtype].itemsize
    _require(byte_count <= MAX_PAYLOAD_BYTES, "tensor exceeds payload limit")
    return byte_count


@dataclass(frozen=True)
class Tensor:
    """An owned immutable, little-endian tensor; bf16 data contains raw bit patterns."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    def __post_init__(self) -> None:
        expected = _tensor_size(self.dtype, self.shape)
        _require(
            type(self.data) is bytes and len(self.data) == expected,
            "invalid tensor bytes",
        )

    @classmethod
    def from_array(cls, value: np.ndarray, *, dtype: str | None = None) -> Tensor:
        """Copy a bounded native array; floating values may include -Inf only."""
        _require(isinstance(value, np.ndarray), "tensor source must be a NumPy array")
        _require(value.nbytes <= MAX_PAYLOAD_BYTES, "source tensor exceeds limit")
        if dtype is None:
            dtype = next(
                (
                    name
                    for name, native in _DTYPES.items()
                    if name != "bf16" and native == value.dtype.newbyteorder("<")
                ),
                None,
            )
        if dtype is None:
            raise GenerationArtifactFormatError(
                f"unsupported source dtype {value.dtype}"
            )
        if dtype == "bf16":
            raise GenerationArtifactFormatError("use raw Tensor bytes for bf16")
        _tensor_size(dtype, tuple(value.shape))
        target = _DTYPES[dtype]
        if dtype in _INTEGER_DTYPES:
            _require(value.dtype.kind in "iu", "integer tensor requires integer source")
            bounds = np.iinfo(target)
            _require(
                not value.size
                or int(value.min()) >= bounds.min
                and int(value.max()) <= bounds.max,
                "integer conversion out of range",
            )
        else:
            _require(
                value.dtype.kind == "f", "floating tensor requires floating source"
            )
            if value.size:
                low, high = float(value.min()), float(value.max())
                _require(
                    not (np.isnan(low) or np.isnan(high) or high == np.inf),
                    "floating source contains unsupported nonfinite value",
                )
                if low == -np.inf:
                    finite = value[value != -np.inf]
                    low = float(finite.min()) if finite.size else 0.0
                bound = float(np.finfo(target).max)
                _require(
                    low >= -bound and high <= bound,
                    "floating conversion out of range",
                )
        return cls(
            dtype,
            tuple(value.shape),
            value.astype(target, copy=False).tobytes(order="C"),
        )

    def as_array(self) -> np.ndarray:
        """Return a read-only view; bf16 is exposed as raw uint16 bit patterns."""
        return np.frombuffer(self.data, dtype=_DTYPES[self.dtype]).reshape(self.shape)


@dataclass(frozen=True)
class RouterRoutes:
    router_id: int
    expert_count: int
    expert_ids: Tensor


@dataclass(frozen=True)
class MoeRoutes:
    token_start: int
    routers: tuple[RouterRoutes, ...]


@dataclass(frozen=True)
class SelectedLogprobs:
    token_start: int
    logprobs: Tensor


@dataclass(frozen=True)
class TopKLogprobs:
    token_start: int
    candidate_token_ids: Tensor
    logprobs: Tensor


GenerationComponent = MoeRoutes | SelectedLogprobs | TopKLogprobs


@dataclass(frozen=True)
class GenerationArtifactChoice:
    choice_index: int
    prompt_token_count: int
    sequence_token_ids: Tensor
    components: tuple[GenerationComponent, ...] = ()


@dataclass(frozen=True)
class GenerationArtifactView:
    choices: tuple[GenerationArtifactChoice, ...]


@dataclass(frozen=True)
class EncodedGenerationArtifact:
    data: bytes
    byte_count: int
    sha256: str


def compact_routes(
    native: np.ndarray,
    router_ids: tuple[int, ...],
    expert_counts: tuple[int, ...],
    *,
    token_start: int,
) -> MoeRoutes:
    """Split native [T,R,K] capture into owned per-router compact wire tensors."""
    _require(
        isinstance(native, np.ndarray) and native.ndim == 3,
        "native routes require rank 3",
    )
    _require(native.nbytes <= MAX_PAYLOAD_BYTES, "native routes exceed limit")
    _require(
        len(router_ids) == len(expert_counts) == native.shape[1],
        "router geometry mismatch",
    )
    _require(0 < len(router_ids) <= _MAX_ROUTERS, "invalid router count")
    _integer(token_start, 0, _MAX_DIMENSION, "route start")
    routers = tuple(
        RouterRoutes(
            router_id,
            count,
            Tensor.from_array(native[:, index, :], dtype=_route_dtype(count)),
        )
        for index, (router_id, count) in enumerate(zip(router_ids, expert_counts))
    )
    component = MoeRoutes(token_start, routers)
    _validate_routes(component, native.shape[0], strict_dtype=True)
    return component


def _route_dtype(expert_count: int) -> str:
    _integer(expert_count, 1, 65536, "expert count")
    return "u8" if expert_count <= 256 else "u16"


def _validate_routes(
    component: MoeRoutes, rows: int, *, strict_dtype: bool
) -> MoeRoutes:
    _require(
        isinstance(component.routers, tuple)
        and 0 < len(component.routers) <= _MAX_ROUTERS,
        "invalid router count",
    )
    ids = tuple(
        _integer(router.router_id, 0, (1 << 63) - 1, "router ID")
        for router in component.routers
    )
    _require(len(set(ids)) == len(ids), "duplicate router ID")
    routers = []
    for router in component.routers:
        dtype = _route_dtype(router.expert_count)
        tensor = router.expert_ids
        _require(
            tensor.dtype in _INTEGER_DTYPES and len(tensor.shape) == 2,
            "routes require integer [T,K_r]",
        )
        _require(
            tensor.shape[0] == rows and 1 <= tensor.shape[1] <= router.expert_count,
            "route geometry mismatch",
        )
        values = tensor.as_array()
        _require(
            not values.size
            or int(values.min()) >= 0
            and int(values.max()) < router.expert_count,
            "expert ID outside router range",
        )
        if values.shape[1] > 1:
            ordered = np.sort(values, axis=1)
            _require(
                not np.any(ordered[:, 1:] == ordered[:, :-1]),
                "duplicate expert ID within row",
            )
        if strict_dtype:
            _require(tensor.dtype == dtype, "route dtype does not match expert count")
        elif tensor.dtype != dtype:
            tensor = Tensor.from_array(values, dtype=dtype)
        routers.append(replace(router, expert_ids=tensor))
    return replace(component, routers=tuple(routers))


def _validate_component(
    component: GenerationComponent, sequence: int, prompt: int, *, strict_dtype: bool
) -> GenerationComponent:
    if isinstance(component, MoeRoutes):
        extent = sequence - 1 if sequence > prompt else sequence
        start = _integer(component.token_start, 0, extent, "route start")
        return _validate_routes(component, extent - start, strict_dtype=strict_dtype)
    _require(
        isinstance(component, (SelectedLogprobs, TopKLogprobs)), "unsupported component"
    )
    tensor = component.logprobs
    _require(tensor.dtype in _FLOAT_DTYPES, "logprobs require floating dtype")
    values = tensor.as_array()
    if tensor.dtype == "bf16":
        allowed = ((values & 0x7F80) != 0x7F80) | (values == 0xFF80)
    else:
        allowed = np.isfinite(values) | np.isneginf(values)
    _require(
        bool(np.all(allowed)),
        "invalid logprob value",
    )
    start = _integer(component.token_start, 0, sequence, "logprob start")
    _require(start + tensor.shape[0] <= sequence, "logprob rows exceed sequence")
    if isinstance(component, SelectedLogprobs):
        _require(len(tensor.shape) == 1, "selected logprobs require rank 1")
    else:
        candidates = component.candidate_token_ids
        _require(
            len(tensor.shape) == 2 and 1 <= tensor.shape[1] <= _MAX_TOPK,
            "invalid top-k shape",
        )
        _require(
            candidates.dtype in _INTEGER_DTYPES and candidates.shape == tensor.shape,
            "top-k candidate shape mismatch",
        )
        _require(
            not candidates.as_array().size or int(candidates.as_array().min()) >= 0,
            "negative candidate token ID",
        )
    return component


def _validated(
    view: GenerationArtifactView, *, strict_dtype: bool
) -> GenerationArtifactView:
    _require(
        isinstance(view.choices, tuple) and 0 < len(view.choices) <= _MAX_CHOICES,
        "invalid choice count",
    )
    _capture_bounds(view)
    choices = []
    previous = -1
    for choice in view.choices:
        index = _integer(choice.choice_index, 0, (1 << 31) - 1, "choice index")
        _require(index > previous, "choices must be unique and ordered")
        previous = index
        tokens = choice.sequence_token_ids
        _require(
            tokens.dtype in _INTEGER_DTYPES and len(tokens.shape) == 1,
            "sequence tokens require integer rank 1",
        )
        _require(
            not tokens.as_array().size or int(tokens.as_array().min()) >= 0,
            "negative sequence token ID",
        )
        sequence = tokens.shape[0]
        prompt = _integer(choice.prompt_token_count, 0, sequence, "prompt token count")
        _require(
            isinstance(choice.components, tuple) and len(choice.components) <= 3,
            "invalid components",
        )
        _require(
            len({type(item) for item in choice.components}) == len(choice.components),
            "duplicate component kind",
        )
        components = tuple(
            _validate_component(item, sequence, prompt, strict_dtype=strict_dtype)
            for item in choice.components
        )
        choices.append(replace(choice, components=components))
    return GenerationArtifactView(tuple(choices))


def _capture_bounds(view: GenerationArtifactView) -> None:
    payload_bytes = 0
    tensor_count = 0
    for choice in view.choices:
        _require(
            isinstance(choice.components, tuple) and len(choice.components) <= 3,
            "invalid components",
        )
        tensors = [choice.sequence_token_ids]
        for component in choice.components:
            if isinstance(component, MoeRoutes):
                _require(
                    isinstance(component.routers, tuple)
                    and 0 < len(component.routers) <= _MAX_ROUTERS,
                    "invalid router count",
                )
                tensors.extend(router.expert_ids for router in component.routers)
            elif isinstance(component, SelectedLogprobs):
                tensors.append(component.logprobs)
            elif isinstance(component, TopKLogprobs):
                tensors.extend((component.candidate_token_ids, component.logprobs))
            else:
                raise GenerationArtifactFormatError("unsupported component")
        payload_bytes += sum(len(tensor.data) for tensor in tensors)
        tensor_count += len(tensors)
        _require(payload_bytes <= MAX_PAYLOAD_BYTES, "capture payload exceeds limit")
        _require(tensor_count <= _MAX_TENSORS, "tensor count exceeds limit")


def _manifest(view: GenerationArtifactView) -> tuple[bytes, tuple[Tensor, ...]]:
    tensors: list[Tensor] = []
    offset = 0

    def reference(tensor: Tensor) -> dict[str, Any]:
        nonlocal offset
        _require(
            len(tensor.data) <= MAX_PAYLOAD_BYTES - offset,
            "artifact payload exceeds limit",
        )
        result = {
            "dtype": tensor.dtype,
            "shape": tensor.shape,
            "offset": offset,
            "byte_count": len(tensor.data),
        }
        offset += len(tensor.data)
        tensors.append(tensor)
        return result

    choices = []
    for choice in view.choices:
        tokens = reference(choice.sequence_token_ids)
        components = []
        for component in choice.components:
            base = {"token_start": component.token_start}
            if isinstance(component, MoeRoutes):
                components.append(
                    {
                        **base,
                        "kind": "moe_routes",
                        "routers": [
                            {
                                "router_id": router.router_id,
                                "expert_count": router.expert_count,
                                "expert_ids": reference(router.expert_ids),
                            }
                            for router in component.routers
                        ],
                    }
                )
            elif isinstance(component, SelectedLogprobs):
                components.append(
                    {
                        **base,
                        "kind": "selected_logprobs",
                        "logprobs": reference(component.logprobs),
                    }
                )
            else:
                components.append(
                    {
                        **base,
                        "kind": "topk_logprobs",
                        "candidate_token_ids": reference(component.candidate_token_ids),
                        "logprobs": reference(component.logprobs),
                    }
                )
        choices.append(
            {
                "choice_index": choice.choice_index,
                "prompt_token_count": choice.prompt_token_count,
                "sequence_token_ids": tokens,
                "components": components,
            }
        )
    encoded = msgspec.msgpack.encode({"choices": choices}, order="deterministic")
    _require(len(encoded) <= MAX_MANIFEST_BYTES, "manifest exceeds limit")
    return encoded, tuple(tensors)


def iter_generation_artifact(
    view: GenerationArtifactView,
    *,
    codec: int = CODEC_ZSTD,
    chunk_size: int = 65536,
) -> Iterator[bytes]:
    """Yield bounded chunks without concatenating an uncompressed body."""
    _require(
        type(codec) is int and codec in (CODEC_NONE, CODEC_ZSTD), "unsupported codec"
    )
    _integer(chunk_size, 1, 1 << 20, "chunk size")
    manifest, tensors = _manifest(_validated(view, strict_dtype=False))
    payload_bytes = sum(len(tensor.data) for tensor in tensors)
    padding = bytes(-len(manifest) % 64)
    decoded_size = len(manifest) + len(padding) + payload_bytes
    yield _PRELUDE.pack(b"DYNEXP1\0", 1, 0, codec, len(manifest), payload_bytes)
    compressor = (
        zstd.ZstdCompressor(level=1, write_content_size=True).compressobj(
            size=decoded_size
        )
        if codec == CODEC_ZSTD
        else None
    )
    for section in (manifest, padding, *(tensor.data for tensor in tensors)):
        for offset in range(0, len(section), chunk_size):
            chunk = section[offset : offset + chunk_size]
            encoded = compressor.compress(chunk) if compressor is not None else chunk
            if encoded:
                yield encoded
    if compressor is not None:
        yield compressor.flush()


def encode_generation_artifact(
    view: GenerationArtifactView, *, codec: int = CODEC_ZSTD
) -> EncodedGenerationArtifact:
    """Collect the stream under a hard object limit for inline and single-PUT use."""
    digest = hashlib.sha256()
    with io.BytesIO() as output:
        for chunk in iter_generation_artifact(view, codec=codec):
            _require(
                output.tell() + len(chunk) <= MAX_OBJECT_BYTES,
                "encoded artifact exceeds limit",
            )
            output.write(chunk)
            digest.update(chunk)
        data = output.getvalue()
    return EncodedGenerationArtifact(data, len(data), digest.hexdigest())


def _keys(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == expected, f"invalid {name} fields")
    return value


def _list(value: Any, maximum: int, name: str, *, nonempty: bool = False) -> list[Any]:
    _require(
        type(value) is list and int(nonempty) <= len(value) <= maximum,
        f"invalid {name} count",
    )
    return value


def _parse_manifest(manifest: Any, payload: bytes) -> GenerationArtifactView:
    _keys(manifest, {"choices"}, "manifest")
    offset = 0
    tensor_count = 0

    def tensor(raw: Any) -> Tensor:
        nonlocal offset, tensor_count
        tensor_count += 1
        _require(tensor_count <= _MAX_TENSORS, "tensor count exceeds limit")
        item = _keys(raw, {"dtype", "shape", "offset", "byte_count"}, "tensor")
        shape = tuple(_list(item["shape"], 2, "dimensions", nonempty=True))
        size = _tensor_size(item["dtype"], shape)
        _integer(item["offset"], 0, len(payload), "tensor offset")
        _integer(item["byte_count"], 0, len(payload), "tensor byte count")
        _require(
            item["offset"] == offset
            and item["byte_count"] == size
            and offset + size <= len(payload),
            "invalid tensor range",
        )
        result = Tensor(item["dtype"], shape, payload[offset : offset + size])
        offset += size
        return result

    def component(raw: Any) -> GenerationComponent:
        _require(type(raw) is dict, "invalid component")
        kind = raw.get("kind")
        if kind == "moe_routes":
            _keys(raw, {"kind", "token_start", "routers"}, "routes")
            routers = []
            for router in _list(raw["routers"], _MAX_ROUTERS, "routers", nonempty=True):
                _keys(router, {"router_id", "expert_count", "expert_ids"}, "router")
                routers.append(
                    RouterRoutes(
                        router["router_id"],
                        router["expert_count"],
                        tensor(router["expert_ids"]),
                    )
                )
            return MoeRoutes(raw["token_start"], tuple(routers))
        if kind == "selected_logprobs":
            _keys(raw, {"kind", "token_start", "logprobs"}, "selected logprobs")
            return SelectedLogprobs(raw["token_start"], tensor(raw["logprobs"]))
        _require(kind == "topk_logprobs", "unsupported component kind")
        _keys(
            raw,
            {"kind", "token_start", "candidate_token_ids", "logprobs"},
            "top-k logprobs",
        )
        return TopKLogprobs(
            raw["token_start"],
            tensor(raw["candidate_token_ids"]),
            tensor(raw["logprobs"]),
        )

    choices = []
    for raw in _list(manifest["choices"], _MAX_CHOICES, "choices", nonempty=True):
        _keys(
            raw,
            {"choice_index", "prompt_token_count", "sequence_token_ids", "components"},
            "choice",
        )
        tokens = tensor(raw["sequence_token_ids"])
        components = tuple(
            component(item) for item in _list(raw["components"], 3, "components")
        )
        choices.append(
            GenerationArtifactChoice(
                raw["choice_index"], raw["prompt_token_count"], tokens, components
            )
        )
    _require(offset == len(payload), "tensor ranges do not cover payload")
    return _validated(GenerationArtifactView(tuple(choices)), strict_dtype=True)


def decode_generation_artifact(
    data: bytes | bytearray | memoryview,
    *,
    expected_router_profile: tuple[tuple[int, int], ...] | None = None,
) -> GenerationArtifactView:
    """Validate the object, optionally binding ordered router IDs and expert counts."""
    try:
        source = memoryview(data)
    except (TypeError, ValueError):
        raise GenerationArtifactFormatError("invalid object buffer") from None
    try:
        _require(
            source.ndim == 1 and source.itemsize == 1 and source.c_contiguous,
            "invalid object buffer",
        )
        _require(
            _PRELUDE.size <= source.nbytes <= MAX_OBJECT_BYTES,
            "invalid object size",
        )
        if type(data) is not bytes:
            data = source.tobytes()
    finally:
        source.release()
    magic, major, minor, codec, manifest_bytes, payload_bytes = _PRELUDE.unpack_from(
        data
    )
    _require(
        magic == b"DYNEXP1\0" and (major, minor) == (1, 0),
        "unsupported artifact version",
    )
    _require(codec in (CODEC_NONE, CODEC_ZSTD), "unsupported codec")
    _require(
        0 < manifest_bytes <= MAX_MANIFEST_BYTES and payload_bytes <= MAX_PAYLOAD_BYTES,
        "declared sizes exceed limits",
    )
    payload_start = (manifest_bytes + 63) // 64 * 64
    decoded_size = payload_start + payload_bytes
    encoded = data[_PRELUDE.size :]
    if codec == CODEC_ZSTD:
        try:
            frame = zstd.get_frame_parameters(encoded)
            _require(
                frame.dict_id == 0
                and frame.content_size == decoded_size
                and frame.window_size <= MAX_PAYLOAD_BYTES + MAX_MANIFEST_BYTES + 63,
                "invalid zstd frame parameters",
            )
            body = zstd.ZstdDecompressor().decompress(
                encoded, max_output_size=decoded_size, allow_extra_data=False
            )
        except zstd.ZstdError as exc:
            raise GenerationArtifactFormatError("invalid zstd frame") from exc
    else:
        body = encoded
    _require(len(body) == decoded_size, "decoded body size mismatch")
    _require(not any(body[manifest_bytes:payload_start]), "nonzero alignment padding")
    manifest_data = body[:manifest_bytes]
    try:
        manifest = msgspec.msgpack.decode(manifest_data)
        canonical = msgspec.msgpack.encode(manifest, order="deterministic")
    except (msgspec.DecodeError, TypeError, ValueError, RecursionError) as exc:
        raise GenerationArtifactFormatError("invalid MessagePack manifest") from exc
    _require(canonical == manifest_data, "noncanonical or duplicate MessagePack keys")
    view = _parse_manifest(manifest, body[payload_start:])
    if expected_router_profile is not None:
        for choice in view.choices:
            for component in choice.components:
                if isinstance(component, MoeRoutes):
                    _require(
                        tuple(
                            (router.router_id, router.expert_count)
                            for router in component.routers
                        )
                        == expected_router_profile,
                        "router profile differs from model capability",
                    )
    return view
