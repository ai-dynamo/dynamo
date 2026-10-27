# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Portable Qwen3-VL inputs for KV reuse and disaggregated recomputation."""

import asyncio
import json
import logging
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any

import torch
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldConfig,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)

from dynamo.common.multimodal.mm_kwargs_transfer import (
    MM_NIXL_CLEANUP_TIMEOUT_S,
    MmKwargsNixlReceiver,
    MmKwargsNixlSender,
    MmKwargsTransferMetadata,
    decode_mm_kwargs_item,
)

logger = logging.getLogger(__name__)

QWEN_VIDEO_HANDOFF_KEY = "qwen_video_input"
_MAX_HANDOFF_BYTES = 2 * 1024 * 1024
_MAX_METADATA_CACHE_BYTES = 8 * 1024 * 1024
_METADATA_FIELDS = {
    "image": ("image_grid_thw",),
    "video": ("video_grid_thw", "timestamps"),
}
_ENCODER_FIELDS = {
    "image": ("pixel_values", "image_embeds"),
    "video": ("pixel_values_videos", "video_embeds"),
}


@dataclass
class _Handoff:
    version: int
    prompt_token_ids: list[int]
    mm_hashes: dict[str, list[str]]
    mm_placeholders: dict[str, list[dict[str, Any]]]
    metadata: dict[str, list[dict[str, Any]]]


@dataclass
class PreparedVideoHandoff:
    """Request-scoped inputs registered until decode completes its READ."""

    state: dict[str, Any]
    owner: "QwenVideoHandoff"
    cleanup_items: list[Any]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        # Decode is dispatched only after the prefill stream closes. Waiting
        # for its READ here would deadlock that dispatch.
        self.owner.release(self.cleanup_items)


class QwenVideoHandoff:
    """Transfer native video inputs and preserve their exact prompt layout."""

    def __init__(self) -> None:
        self._metadata: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._cache_bytes = 0
        self._sender: MmKwargsNixlSender | None = None
        self._receiver: MmKwargsNixlReceiver | None = None
        self._cleanup_tasks: set[asyncio.Task] = set()

    def release(self, items: list[Any]) -> None:
        """Defer bounded transport cleanup until decode has read the inputs."""
        if not items:
            return
        task = asyncio.create_task(self._sender.cleanup(items))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_done)

    def _cleanup_done(self, task: asyncio.Task) -> None:
        self._cleanup_tasks.discard(task)
        error = None if task.cancelled() else task.exception()
        if error is not None:
            # Cleanup runs after the response; it cannot fail that request.
            logger.error(
                "Video input transfer cleanup failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def prepare(self, processed: dict[str, Any]) -> PreparedVideoHandoff | None:
        """Stage full native inputs so decode can also survive KV recomputation."""
        layout = self.capture(processed)
        if layout is None or any(
            item is None
            or not any(
                name in item and item[name].data is not None
                for name in _ENCODER_FIELDS[modality]
            )
            for modality, items in processed["mm_kwargs"].items()
            for item in items
        ):
            return None
        if self._sender is None:
            self._sender = MmKwargsNixlSender()
        transfers = {}
        cleanup_items = []
        try:
            for modality, hashes in processed["mm_hashes"].items():
                features = [
                    MultiModalFeatureSpec(
                        data=item,
                        modality=modality,
                        identifier=identity,
                        mm_hash=identity,
                        mm_position=span,
                    )
                    for identity, item, span in zip(
                        hashes,
                        processed["mm_kwargs"][modality],
                        processed["mm_placeholders"][modality],
                        strict=True,
                    )
                ]
                extra_args, items = await self._sender.prepare(features, modality)
                cleanup_items.extend(items)
                if not extra_args:
                    return None
                transfers[modality] = extra_args["mm_kwargs_nixl"]
            state = {"version": 2, "layout": layout, "inputs": transfers}
            if len(json.dumps(state, allow_nan=False).encode()) > _MAX_HANDOFF_BYTES:
                return None
            handoff = PreparedVideoHandoff(state, self, cleanup_items)
            cleanup_items = []
            return handoff
        finally:
            self.release(cleanup_items)

    async def restore_for_decode(self, state: Any, **kwargs) -> dict[str, Any] | None:
        """Receive native inputs before submission, including recomputation data."""
        if not isinstance(state, dict) or state.get("version") != 2:
            # Version 1 has no pixels/embeddings and cannot survive preemption.
            self.restore(state, **kwargs)
            return None
        if type(state["version"]) is not int:
            raise ValueError("Invalid Qwen video handoff version")
        if len(json.dumps(state, allow_nan=False).encode()) > _MAX_HANDOFF_BYTES:
            raise ValueError("Qwen video handoff exceeds the metadata limit")
        restored = self.restore(state["layout"], **kwargs)
        if restored is None:
            return None
        transfers = state["inputs"]
        if set(transfers) != set(restored["mm_hashes"]):
            raise ValueError("Video input transfer modalities do not match")
        metadata = {}
        for modality, raw in transfers.items():
            transfer = MmKwargsTransferMetadata.model_validate(raw)
            if (
                transfer.modality != modality
                or transfer.mm_hashes != restored["mm_hashes"][modality]
                or len(transfer.tensor_specs) != len(transfer.mm_hashes)
                or any(
                    spec.field_name != "__pickled_kwargs_item__"
                    for spec in transfer.tensor_specs
                )
            ):
                raise ValueError("Video input transfer items do not match the layout")
            metadata[modality] = transfer
        if self._receiver is None:
            self._receiver = MmKwargsNixlReceiver()
        for modality, transfer in metadata.items():
            received = await asyncio.wait_for(
                self._receiver.receive(transfer), timeout=MM_NIXL_CLEANUP_TIMEOUT_S
            )
            payloads = received["__pickled_kwargs_item__"]
            if len(payloads) != len(transfer.mm_hashes):
                raise ValueError("Video input transfer returned the wrong item count")
            items = [decode_mm_kwargs_item(payload) for payload in payloads]
            for item, expected in zip(
                items, restored["mm_kwargs"][modality], strict=True
            ):
                if not any(
                    name in item and item[name].data is not None
                    for name in _ENCODER_FIELDS[modality]
                ):
                    raise ValueError("Video input transfer is missing encoder inputs")
                for name, field in expected.items():
                    value = torch.as_tensor(item[name].data, dtype=field.data.dtype)
                    if not torch.equal(value, field.data):
                        raise ValueError("Video input transfer metadata does not match")
            restored["mm_kwargs"][modality] = items
        return restored

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
        """Validate the layout; engine submission also requires the full inputs."""
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
