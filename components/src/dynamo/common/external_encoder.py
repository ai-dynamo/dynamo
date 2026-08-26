# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned MsgPack handoff from an external encoder to an LLM worker."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from types import MappingProxyType
from typing import Any, Mapping

import torch

EXTERNAL_ENCODER_RESULT_SCHEMA = "dynamo.external_encoder_result"
EXTERNAL_ENCODER_RESULT_VERSION = 0
LINEAR_EMBEDDINGS_FORMAT = "linear_embeddings"
REQUEST_PLANE_TRANSPORT = "request_plane_msgpack"

_SUPPORTED_DTYPES: Mapping[str, torch.dtype] = MappingProxyType(
    {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
)


def _check_keys(data: Mapping[str, Any], required: set[str], kind: str) -> None:
    missing = required - set(data)
    unknown = set(data) - required
    if missing:
        raise ValueError(f"{kind} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{kind} has unknown fields: {sorted(unknown)}")


def encode_request_plane_tensor(tensor: torch.Tensor) -> dict[str, Any]:
    """Encode one owned CPU tensor into a MsgPack request-plane payload.

    The binary ``data`` field deliberately avoids base64 overhead. The target
    endpoint must therefore advertise ``DYN_REQUEST_PLANE_CODEC=msgpack``.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(
            "external encoder features must be a torch.Tensor; "
            f"got {type(tensor).__name__}"
        )
    if tensor.device.type != "cpu":
        raise ValueError(f"external encoder tensor is on {tensor.device}; expected CPU")
    if not tensor.is_contiguous():
        raise ValueError("external encoder tensor must be contiguous")
    if tensor.dim() != 2 or any(dimension <= 0 for dimension in tensor.shape):
        raise ValueError(
            "external encoder tensor must be a non-empty 2D tensor; "
            f"got shape {tuple(tensor.shape)}"
        )
    dtype = str(tensor.dtype).removeprefix("torch.")
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported external encoder tensor dtype {tensor.dtype}")

    raw = tensor.detach().view(torch.uint8).numpy().tobytes()
    return {
        "transport": REQUEST_PLANE_TRANSPORT,
        "shape": list(tensor.shape),
        "dtype": dtype,
        "data": raw,
    }


def decode_request_plane_tensor(payload: Mapping[str, Any]) -> torch.Tensor:
    """Decode an owned CPU tensor from Dynamo's ordinary request payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("external encoder features must be an object")
    _check_keys(
        payload,
        {"transport", "shape", "dtype", "data"},
        "external encoder features",
    )
    if payload["transport"] != REQUEST_PLANE_TRANSPORT:
        raise ValueError(
            f"unsupported external encoder transport {payload['transport']!r}"
        )

    shape = payload["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in shape
        )
    ):
        raise ValueError(
            "external encoder tensor shape must contain two positive integers"
        )
    dtype_name = payload["dtype"]
    if not isinstance(dtype_name, str) or dtype_name not in _SUPPORTED_DTYPES:
        raise ValueError(f"unsupported external encoder tensor dtype {dtype_name!r}")
    data = payload["data"]
    if not isinstance(data, bytes):
        raise ValueError("external encoder tensor data must be bytes")

    dtype = _SUPPORTED_DTYPES[dtype_name]
    expected_bytes = prod(shape) * torch.empty((), dtype=dtype).element_size()
    if len(data) != expected_bytes:
        raise ValueError(
            "external encoder tensor byte count does not match shape and dtype; "
            f"expected {expected_bytes}, got {len(data)}"
        )

    storage = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return storage.view(dtype).reshape(shape)


@dataclass(frozen=True)
class ExternalEncoderResult:
    """Packed linear embeddings plus the metadata needed to splice them."""

    features: Mapping[str, Any]
    row_splits: tuple[int, ...]
    image_token_id: int
    format: str = LINEAR_EMBEDDINGS_FORMAT

    def __post_init__(self) -> None:
        if not isinstance(self.features, Mapping) or not self.features:
            raise ValueError("external encoder features must be a non-empty object")
        if self.format != LINEAR_EMBEDDINGS_FORMAT:
            raise ValueError(f"unsupported external encoder format {self.format!r}")
        row_splits = tuple(self.row_splits)
        if len(row_splits) < 2 or row_splits[0] != 0:
            raise ValueError("external encoder row_splits must start at zero")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in row_splits
        ):
            raise ValueError(
                "external encoder row_splits must contain non-negative integers"
            )
        if any(left > right for left, right in zip(row_splits, row_splits[1:])):
            raise ValueError("external encoder row_splits must be non-decreasing")
        if isinstance(self.image_token_id, bool) or not isinstance(
            self.image_token_id, int
        ):
            raise ValueError("external encoder image_token_id must be an integer")
        if self.image_token_id < 0:
            raise ValueError("external encoder image_token_id must be non-negative")

        shape = self.features.get("shape")
        if (
            isinstance(shape, list)
            and shape
            and isinstance(shape[0], int)
            and not isinstance(shape[0], bool)
            and row_splits[-1] != shape[0]
        ):
            raise ValueError(
                "external encoder row_splits do not cover the packed feature rows"
            )
        object.__setattr__(self, "features", MappingProxyType(dict(self.features)))
        object.__setattr__(self, "row_splits", row_splits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_ENCODER_RESULT_SCHEMA,
            "version": EXTERNAL_ENCODER_RESULT_VERSION,
            "format": self.format,
            "features": dict(self.features),
            "row_splits": list(self.row_splits),
            "image_token_id": self.image_token_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExternalEncoderResult":
        if not isinstance(data, Mapping):
            raise ValueError("external encoder result must be an object")
        _check_keys(
            data,
            {
                "schema",
                "version",
                "format",
                "features",
                "row_splits",
                "image_token_id",
            },
            "external encoder result",
        )
        if data["schema"] != EXTERNAL_ENCODER_RESULT_SCHEMA:
            raise ValueError(f"unsupported external encoder schema {data['schema']!r}")
        version = data["version"]
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or version != EXTERNAL_ENCODER_RESULT_VERSION
        ):
            raise ValueError(f"unsupported external encoder version {version!r}")
        row_splits = data["row_splits"]
        if not isinstance(row_splits, list):
            raise ValueError("external encoder row_splits must be an array")
        return cls(
            features=data["features"],
            row_splits=tuple(row_splits),
            image_token_id=data["image_token_id"],
            format=data["format"],
        )

    @classmethod
    def from_parts(
        cls, features: Mapping[str, Any], metadata: Mapping[str, Any]
    ) -> "ExternalEncoderResult":
        if not isinstance(metadata, Mapping):
            raise ValueError("external encoder metadata must be an object")
        _check_keys(
            metadata,
            {"row_splits", "image_token_id"},
            "external encoder metadata",
        )
        row_splits = metadata["row_splits"]
        if not isinstance(row_splits, list):
            raise ValueError("external encoder row_splits must be an array")
        return cls(
            features=features,
            row_splits=tuple(row_splits),
            image_token_id=metadata["image_token_id"],
        )
