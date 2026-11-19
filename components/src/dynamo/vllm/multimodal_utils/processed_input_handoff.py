# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer native vLLM video inputs without interpreting model-specific fields."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt, StrictStr
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

from dynamo.common.multimodal.mm_kwargs_transfer import (
    MM_NIXL_CLEANUP_TIMEOUT_S,
    MmKwargsNixlReceiver,
    MmKwargsNixlSender,
    MmKwargsTransferMetadata,
    decode_mm_kwargs_item,
)

logger = logging.getLogger(__name__)

VIDEO_HANDOFF_KEY = "processed_video_input"
_VERSION = 3
_MAX_HANDOFF_BYTES = 2 * 1024 * 1024


class _Placeholder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    offset: StrictInt = Field(ge=0)
    length: StrictInt = Field(gt=0)
    is_embed: list[StrictBool] | None


class _Layout(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_token_ids: list[StrictInt] = Field(min_length=1)
    mm_hashes: dict[str, list[StrictStr]]
    mm_placeholders: dict[str, list[_Placeholder]]
    field_names: dict[str, list[list[StrictStr]]]
    cache_salt: StrictStr | None = None
    prompt: StrictStr | None = None


@dataclass
class PreparedInputHandoff:
    """Keep registered inputs alive until decode has completed its READ."""

    state: dict[str, Any]
    owner: "ProcessedInputHandoff"
    cleanup_items: list[Any]

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        # Decode is dispatched after prefill closes its stream.
        self.owner.release(self.cleanup_items)


class ProcessedInputHandoff:
    """Preserve complete native inputs for both KV reuse and recomputation."""

    def __init__(self) -> None:
        self._sender: MmKwargsNixlSender | None = None
        self._receiver: MmKwargsNixlReceiver | None = None
        self._cleanup_tasks: set[asyncio.Task] = set()

    def release(self, items: list[Any]) -> None:
        if not items:
            return
        sender = self._sender
        if sender is None:
            raise RuntimeError("Video input transfer sender is not initialized")
        task = asyncio.create_task(sender.cleanup(items))
        self._cleanup_tasks.add(task)
        task.add_done_callback(self._cleanup_done)

    def _cleanup_done(self, task: asyncio.Task) -> None:
        self._cleanup_tasks.discard(task)
        error = None if task.cancelled() else task.exception()
        if error is not None:
            logger.error(
                "Video input transfer cleanup failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    async def prepare(self, processed: dict[str, Any]) -> PreparedInputHandoff | None:
        """Stage full items from a native render with skip_mm_cache=True."""
        if (
            processed.get("type") != "multimodal"
            or "encoder_prompt_token_ids" in processed
        ):
            return None
        hashes = processed["mm_hashes"]
        if "video" not in hashes or set(hashes) - {"image", "video"}:
            return None
        if any(
            item is None or not item
            for items in processed["mm_kwargs"].values()
            for item in items
        ):
            return None
        layout = {
            "prompt_token_ids": processed["prompt_token_ids"],
            "mm_hashes": hashes,
            "mm_placeholders": {
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
            },
            "field_names": {
                modality: [sorted(item) for item in items]
                for modality, items in processed["mm_kwargs"].items()
            },
        }
        for key in ("cache_salt", "prompt"):
            if key in processed:
                layout[key] = processed[key]
        if len(json.dumps(layout, allow_nan=False).encode()) > _MAX_HANDOFF_BYTES:
            return None
        if self._sender is None:
            self._sender = MmKwargsNixlSender()
        transfers = {}
        cleanup_items = []
        try:
            for modality, identities in hashes.items():
                features = [
                    MultiModalFeatureSpec(
                        data=item,
                        modality=modality,
                        identifier=identity,
                        mm_hash=identity,
                        mm_position=span,
                    )
                    for identity, item, span in zip(
                        identities,
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
            state = {"version": _VERSION, "layout": layout, "inputs": transfers}
            if len(json.dumps(state, allow_nan=False).encode()) > _MAX_HANDOFF_BYTES:
                return None
            handoff = PreparedInputHandoff(state, self, cleanup_items)
            cleanup_items = []
            return handoff
        finally:
            self.release(cleanup_items)

    async def restore_for_decode(
        self, state: Any, *, max_model_len: int, media_counts: dict[str, int]
    ) -> dict[str, Any] | None:
        """Validate the native layout and receive every serialized item field."""
        if not isinstance(state, dict) or state.get("version") != _VERSION:
            return None
        if type(state["version"]) is not int:
            raise ValueError("Invalid video input handoff version")
        if set(state) != {"version", "layout", "inputs"}:
            raise ValueError("Invalid video input handoff fields")
        if len(json.dumps(state, allow_nan=False).encode()) > _MAX_HANDOFF_BYTES:
            raise ValueError("Video input handoff exceeds the metadata limit")
        layout = _Layout.model_validate(state["layout"])
        tokens = layout.prompt_token_ids
        if len(tokens) > max_model_len or any(token < 0 for token in tokens):
            raise ValueError("Invalid video input handoff prompt tokens")
        modalities = set(layout.mm_hashes)
        if (
            "video" not in modalities
            or modalities - {"image", "video"}
            or modalities != set(layout.mm_placeholders)
            or modalities != set(layout.field_names)
            or modalities != {key for key, count in media_counts.items() if count}
            or not isinstance(state["inputs"], dict)
            or modalities != set(state["inputs"])
        ):
            raise ValueError("Video input handoff modalities do not match the request")
        intervals = []
        metadata = {}
        placeholders: dict[str, list[PlaceholderRange]] = {}
        for modality, identities in layout.mm_hashes.items():
            spans = layout.mm_placeholders[modality]
            fields = layout.field_names[modality]
            transfer = MmKwargsTransferMetadata.model_validate(
                state["inputs"][modality]
            )
            if (
                not (
                    len(identities)
                    == len(spans)
                    == len(fields)
                    == media_counts[modality]
                )
                or any(not identity for identity in identities)
                or any(not names or len(set(names)) != len(names) for names in fields)
                or transfer.modality != modality
                or transfer.mm_hashes != identities
                or len(transfer.tensor_specs) != len(identities)
                or any(
                    spec.field_name != "__pickled_kwargs_item__"
                    for spec in transfer.tensor_specs
                )
            ):
                raise ValueError("Video input transfer items do not match the layout")
            placeholders[modality] = []
            for span in spans:
                if span.offset + span.length > len(tokens) or (
                    span.is_embed is not None and len(span.is_embed) != span.length
                ):
                    raise ValueError("Invalid video input handoff placeholder")
                intervals.append((span.offset, span.offset + span.length))
                placeholders[modality].append(
                    PlaceholderRange(
                        offset=span.offset,
                        length=span.length,
                        is_embed=(
                            None
                            if span.is_embed is None
                            else torch.tensor(span.is_embed, dtype=torch.bool)
                        ),
                    )
                )
            metadata[modality] = transfer
        ordered = sorted(intervals)
        if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Overlapping video input handoff placeholders")
        if self._receiver is None:
            self._receiver = MmKwargsNixlReceiver()
        kwargs = {}
        for modality, transfer in metadata.items():
            received = await asyncio.wait_for(
                self._receiver.receive(transfer), timeout=MM_NIXL_CLEANUP_TIMEOUT_S
            )
            payloads = received["__pickled_kwargs_item__"]
            if len(payloads) != len(transfer.mm_hashes):
                raise ValueError("Video input transfer returned the wrong item count")
            items = [decode_mm_kwargs_item(payload) for payload in payloads]
            if any(
                set(item) != set(names)
                for item, names in zip(items, layout.field_names[modality], strict=True)
            ):
                raise ValueError("Video input transfer is missing native fields")
            kwargs[modality] = items
        restored = {
            "type": "multimodal",
            "prompt_token_ids": tokens,
            "mm_hashes": layout.mm_hashes,
            "mm_placeholders": placeholders,
            "mm_kwargs": kwargs,
        }
        if layout.cache_salt is not None:
            restored["cache_salt"] = layout.cache_salt
        if layout.prompt is not None:
            restored["prompt"] = layout.prompt
        return restored
