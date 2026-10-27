# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable Qwen3-VL input metadata for KV-backed disaggregated decode."""

import json
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

import torch
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)

QWEN_VIDEO_HANDOFF_KEY = "qwen_video_input"
_MAX_HANDOFF_BYTES = 2 * 1024 * 1024
_MAX_METADATA_CACHE_BYTES = 8 * 1024 * 1024
_METADATA_FIELDS = {
    "image": ("image_grid_thw",),
    "video": ("video_grid_thw", "timestamps"),
}


@dataclass
class _Handoff:
    version: int
    prompt_token_ids: list[int]
    mm_hashes: dict[str, list[str]]
    mm_placeholders: dict[str, list[dict[str, Any]]]
    metadata: dict[str, list[dict[str, Any]]]


class QwenVideoHandoff:
    """Keep small item metadata when vLLM suppresses tensors on cache hits."""

    def __init__(self) -> None:
        self._metadata: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._cache_bytes = 0

    @staticmethod
    def _entry_size(key: tuple[str, str], encoded: bytes) -> int:
        return sum(len(part.encode()) for part in key) + len(encoded)

    def _remember(self, key: tuple[str, str], metadata: dict[str, Any]) -> None:
        encoded = json.dumps(metadata, allow_nan=False).encode()
        entry_size = self._entry_size(key, encoded)
        if entry_size > _MAX_METADATA_CACHE_BYTES:
            return
        previous = self._metadata.pop(key, None)
        if previous is not None:
            self._cache_bytes -= self._entry_size(key, previous)
        self._metadata[key] = encoded
        self._cache_bytes += entry_size
        while self._cache_bytes > _MAX_METADATA_CACHE_BYTES:
            evicted_key, evicted = self._metadata.popitem(last=False)
            self._cache_bytes -= self._entry_size(evicted_key, evicted)

    def capture(self, processed: dict[str, Any]) -> dict[str, Any] | None:
        """Capture native layout without transferring pixels or embeddings."""
        if processed.get("type") != "multimodal":
            return None
        hashes = processed["mm_hashes"]
        if "video" not in hashes or set(hashes) - _METADATA_FIELDS.keys():
            return None
        metadata: dict[str, list[dict[str, Any]]] = {}
        for modality, identities in hashes.items():
            values = []
            for identity, item in zip(
                identities, processed["mm_kwargs"][modality], strict=True
            ):
                key = (modality, identity)
                if item is not None:
                    fields = {}
                    for name in _METADATA_FIELDS[modality]:
                        if name not in item or item[name].data is None:
                            return None
                        value = item[name].data
                        fields[name] = (
                            value.tolist() if isinstance(value, torch.Tensor) else value
                        )
                    self._remember(key, fields)
                elif key in self._metadata:
                    self._metadata.move_to_end(key)
                    fields = json.loads(self._metadata[key])
                else:
                    # The native sender cache can outlive this metadata cache.
                    return None
                values.append(fields)
            metadata[modality] = values
        ranges = {
            modality: [
                {
                    "offset": span.offset,
                    "length": span.length,
                    "is_embed": (
                        None if span.is_embed is None else span.is_embed.tolist()
                    ),
                }
                for span in spans
            ]
            for modality, spans in processed["mm_placeholders"].items()
        }
        state = asdict(
            _Handoff(1, processed["prompt_token_ids"], hashes, ranges, metadata)
        )
        if len(json.dumps(state, allow_nan=False).encode()) > _MAX_HANDOFF_BYTES:
            return None
        return state

    @staticmethod
    def restore(
        state: Any,
        *,
        max_model_len: int,
        media_counts: dict[str, int],
        spatial_merge_size: int,
    ) -> dict[str, Any] | None:
        """Validate a known handoff and construct a metadata-only EngineInput."""
        if not isinstance(state, dict) or state.get("version") != 1:
            return None
        if type(state["version"]) is not int:
            raise ValueError("Invalid Qwen video handoff version")
        if len(json.dumps(state, allow_nan=False).encode()) > _MAX_HANDOFF_BYTES:
            raise ValueError("Qwen video handoff exceeds the metadata limit")
        handoff = _Handoff(**state)
        tokens = handoff.prompt_token_ids
        if (
            not isinstance(tokens, list)
            or not tokens
            or len(tokens) > max_model_len
            or any(type(token) is not int or token < 0 for token in tokens)
        ):
            raise ValueError("Invalid Qwen video handoff prompt tokens")
        modalities = set(handoff.mm_hashes)
        if (
            "video" not in modalities
            or modalities - _METADATA_FIELDS.keys()
            or modalities != set(handoff.metadata)
            or modalities != set(handoff.mm_placeholders)
            or modalities != {key for key, count in media_counts.items() if count}
        ):
            raise ValueError("Qwen video handoff modalities do not match the request")
        mm_kwargs: dict[str, list[MultiModalKwargsItem]] = {}
        placeholders: dict[str, list[PlaceholderRange]] = {}
        intervals = []
        for modality in modalities:
            hashes = handoff.mm_hashes[modality]
            ranges = handoff.mm_placeholders[modality]
            items = handoff.metadata[modality]
            if not (
                isinstance(hashes, list)
                and isinstance(ranges, list)
                and isinstance(items, list)
                and len(hashes) == len(ranges) == len(items) == media_counts[modality]
            ):
                raise ValueError("Qwen video handoff item counts do not match")
            mm_kwargs[modality] = []
            placeholders[modality] = []
            for identity, span, metadata in zip(hashes, ranges, items, strict=True):
                if not isinstance(identity, str) or not identity:
                    raise ValueError("Invalid Qwen video handoff cache identity")
                offset, length, mask = span["offset"], span["length"], span["is_embed"]
                if (
                    type(offset) is not int
                    or type(length) is not int
                    or offset < 0
                    or length < 1
                    or offset + length > len(tokens)
                    or (
                        mask is not None
                        and (
                            not isinstance(mask, list)
                            or len(mask) != length
                            or any(type(value) is not bool for value in mask)
                        )
                    )
                ):
                    raise ValueError("Invalid Qwen video handoff placeholder")
                intervals.append((offset, offset + length))
                placeholders[modality].append(
                    PlaceholderRange(
                        offset=offset,
                        length=length,
                        is_embed=(
                            None
                            if mask is None
                            else torch.tensor(mask, dtype=torch.bool)
                        ),
                    )
                )
                grid_key = f"{modality}_grid_thw"
                grid = metadata[grid_key]
                if (
                    set(metadata) != set(_METADATA_FIELDS[modality])
                    or not isinstance(grid, list)
                    or len(grid) != 3
                    or any(type(value) is not int or value < 1 for value in grid)
                    or (modality == "image" and grid[0] != 1)
                    or grid[0] > max_model_len
                    or grid[1] % spatial_merge_size
                    or grid[2] % spatial_merge_size
                    or grid[1] * grid[2] // spatial_merge_size**2 > max_model_len
                ):
                    raise ValueError("Invalid Qwen video handoff grid")
                fields = {
                    grid_key: MultiModalFieldElem(
                        data=torch.tensor(grid, dtype=torch.long),
                        field=MultiModalFieldConfig.batched(
                            modality, keep_on_cpu=True
                        ).field,
                    )
                }
                if modality == "video":
                    timestamps = metadata["timestamps"]
                    if (
                        not isinstance(timestamps, list)
                        or len(timestamps) != grid[0]
                        or any(
                            type(value) not in (float, int)
                            or not math.isfinite(value)
                            or value < 0
                            for value in timestamps
                        )
                    ):
                        raise ValueError("Invalid Qwen video handoff timestamps")
                    fields["timestamps"] = MultiModalFieldElem(
                        data=torch.tensor(timestamps, dtype=torch.float64),
                        field=MultiModalFieldConfig.batched(
                            "video", keep_on_cpu=True
                        ).field,
                    )
                mm_kwargs[modality].append(MultiModalKwargsItem(fields))
        ordered = sorted(intervals)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Overlapping Qwen video handoff placeholders")
        # NIXL decode recomputes the last prompt token. Reload media if that
        # token needs visual embeddings, which this handoff does not carry.
        if any(
            span.offset + span.length == len(tokens)
            and (span.is_embed is None or bool(span.is_embed[-1]))
            for spans in placeholders.values()
            for span in spans
        ):
            return None
        return {
            "type": "multimodal",
            "prompt_token_ids": tokens,
            "mm_hashes": handoff.mm_hashes,
            "mm_placeholders": placeholders,
            "mm_kwargs": mm_kwargs,
        }
