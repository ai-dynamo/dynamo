# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JSON-safe artifact contract for externally encoded Qwen image features."""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from safetensors import SafetensorError
from safetensors.torch import load, save

EXTERNAL_QWEN_ARTIFACT_FORMAT = "qwen2_vl_projected_grid.v1"
MAX_EXTERNAL_QWEN_ARTIFACT_BYTES = 32 * 1024 * 1024
_MAX_BASE64_LENGTH = ((MAX_EXTERNAL_QWEN_ARTIFACT_BYTES + 2) // 3) * 4
_ARTIFACT_KEYS = frozenset(
    {
        "format",
        "model",
        "prompt_token_ids",
        "image_embeds",
        "image_grid_thw",
    }
)


def _validate_image_embeds(image_embeds: torch.Tensor) -> torch.Tensor:
    if image_embeds.dim() != 2 or min(image_embeds.shape) < 1:
        raise ValueError(
            "image_embeds must be a non-empty 2D tensor; "
            f"got shape {tuple(image_embeds.shape)}"
        )
    if image_embeds.dtype != torch.bfloat16:
        raise ValueError(
            f"image_embeds must use torch.bfloat16; got {image_embeds.dtype}"
        )
    if image_embeds.device.type != "cpu":
        raise ValueError("image_embeds must be on CPU")
    if not image_embeds.is_contiguous():
        raise ValueError("image_embeds must be contiguous")
    if image_embeds.requires_grad:
        raise ValueError("image_embeds must not require gradients")
    if not torch.isfinite(image_embeds).all().item():
        raise ValueError("image_embeds contains NaN or Inf")
    return image_embeds


def serialize_image_embeds(image_embeds: torch.Tensor) -> str:
    """Serialize one validated image-embedding tensor as base64 safetensors."""

    tensor = _validate_image_embeds(image_embeds)
    payload = save({"image_embeds": tensor})
    if len(payload) > MAX_EXTERNAL_QWEN_ARTIFACT_BYTES:
        raise ValueError("serialized image embeddings exceed the 32 MiB artifact limit")
    return base64.b64encode(payload).decode("ascii")


def deserialize_image_embeds(encoded: str) -> torch.Tensor:
    """Decode and validate a base64 safetensors image-embedding payload."""

    if not isinstance(encoded, str) or not encoded:
        raise ValueError("image_embeds must be a non-empty base64 string")
    if len(encoded) > _MAX_BASE64_LENGTH:
        raise ValueError("encoded image embeddings exceed the 32 MiB artifact limit")

    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_embeds is not valid base64") from exc
    if len(payload) > MAX_EXTERNAL_QWEN_ARTIFACT_BYTES:
        raise ValueError("decoded image embeddings exceed the 32 MiB artifact limit")

    try:
        tensors = load(payload)
    except SafetensorError as exc:
        raise ValueError("image_embeds is not a valid safetensors payload") from exc
    if set(tensors) != {"image_embeds"}:
        raise ValueError(
            "safetensors payload must contain only the 'image_embeds' tensor"
        )
    return _validate_image_embeds(tensors["image_embeds"])


def _normalize_token_ids(values: Any) -> tuple[int, ...]:
    if not isinstance(values, list) or not values:
        raise ValueError("prompt_token_ids must be a non-empty list")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in values
    ):
        raise ValueError("prompt_token_ids must contain non-negative integers")
    return tuple(values)


def _normalize_grid(values: Any) -> tuple[tuple[int, int, int], ...]:
    if not isinstance(values, list) or not values:
        raise ValueError("image_grid_thw must be a non-empty list")

    grids: list[tuple[int, int, int]] = []
    for index, grid in enumerate(values):
        if (
            not isinstance(grid, list)
            or len(grid) != 3
            or any(
                not isinstance(value, int) or isinstance(value, bool) for value in grid
            )
            or any(value < 1 for value in grid)
        ):
            raise ValueError(
                f"image_grid_thw[{index}] must contain three positive integers"
            )
        grids.append((grid[0], grid[1], grid[2]))
    return tuple(grids)


@dataclass(frozen=True)
class ExternalQwenArtifact:
    """Projected Qwen image rows and the prompt/grid metadata that consumes them."""

    model: str
    prompt_token_ids: tuple[int, ...]
    image_embeds: str
    image_grid_thw: tuple[tuple[int, int, int], ...]

    @classmethod
    def create(
        cls,
        *,
        model: str,
        prompt_token_ids: Sequence[int],
        image_embeds: torch.Tensor,
        image_grid_thw: Sequence[Sequence[int]],
    ) -> "ExternalQwenArtifact":
        """Create an artifact from an in-memory projected image tensor."""

        return cls.from_dict(
            {
                "format": EXTERNAL_QWEN_ARTIFACT_FORMAT,
                "model": model,
                "prompt_token_ids": list(prompt_token_ids),
                "image_embeds": serialize_image_embeds(image_embeds),
                "image_grid_thw": [list(grid) for grid in image_grid_thw],
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExternalQwenArtifact":
        """Validate a JSON-compatible artifact without decoding its tensor."""

        if not isinstance(payload, Mapping):
            raise TypeError("external Qwen artifact must be an object")
        keys = set(payload)
        missing = _ARTIFACT_KEYS - keys
        extra = keys - _ARTIFACT_KEYS
        if missing:
            raise ValueError(
                f"external Qwen artifact is missing fields: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"external Qwen artifact has unsupported fields: {sorted(extra)}"
            )
        if payload["format"] != EXTERNAL_QWEN_ARTIFACT_FORMAT:
            raise ValueError(
                "external Qwen artifact format must be "
                f"{EXTERNAL_QWEN_ARTIFACT_FORMAT!r}"
            )

        model = payload["model"]
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty string")
        encoded = payload["image_embeds"]
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("image_embeds must be a non-empty base64 string")
        if len(encoded) > _MAX_BASE64_LENGTH:
            raise ValueError(
                "encoded image embeddings exceed the 32 MiB artifact limit"
            )

        return cls(
            model=model,
            prompt_token_ids=_normalize_token_ids(payload["prompt_token_ids"]),
            image_embeds=encoded,
            image_grid_thw=_normalize_grid(payload["image_grid_thw"]),
        )

    def load_image_embeds(self) -> torch.Tensor:
        """Decode the projected image rows."""

        return deserialize_image_embeds(self.image_embeds)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible wire representation."""

        return {
            "format": EXTERNAL_QWEN_ARTIFACT_FORMAT,
            "model": self.model,
            "prompt_token_ids": list(self.prompt_token_ids),
            "image_embeds": self.image_embeds,
            "image_grid_thw": [list(grid) for grid in self.image_grid_thw],
        }
