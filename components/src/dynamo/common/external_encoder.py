# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned handoff from an external encoder to an LLM worker."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

EXTERNAL_ENCODER_RESULT_SCHEMA = "dynamo.external_encoder_result"
EXTERNAL_ENCODER_RESULT_VERSION = 0
LINEAR_EMBEDDINGS_FORMAT = "linear_embeddings"


def _check_keys(data: Mapping[str, Any], required: set[str]) -> None:
    missing = required - set(data)
    unknown = set(data) - required
    if missing:
        raise ValueError(f"external encoder result missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(
            f"external encoder result has unknown fields: {sorted(unknown)}"
        )


@dataclass(frozen=True)
class ExternalEncoderResult:
    """Transport reference plus metadata for packed linear embeddings."""

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
        _check_keys(metadata, {"row_splits", "image_token_id"})
        row_splits = metadata["row_splits"]
        if not isinstance(row_splits, list):
            raise ValueError("external encoder row_splits must be an array")
        return cls(
            features=features,
            row_splits=tuple(row_splits),
            image_token_id=metadata["image_token_id"],
        )
