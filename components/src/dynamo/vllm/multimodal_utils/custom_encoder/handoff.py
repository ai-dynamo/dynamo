# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned visual-only handoff for linear custom-encoder prompts."""

from __future__ import annotations

import logging
import uuid
from collections import deque
from collections.abc import Mapping
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from vllm.inputs import EmbedsPrompt

from dynamo.common.multimodal.embedding_transfer import (
    AbstractEmbeddingReceiver,
    AbstractEmbeddingSender,
    TransferRequest,
    torch_dtype_to_string,
)
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.linear import (
    LinearVisualPrompt,
)
from dynamo.vllm.multimodal_utils.model_config import _hidden_size

LINEAR_VISUAL_HANDOFF_FORMAT = "linear_visual_embeds"
LINEAR_VISUAL_HANDOFF_VERSION = 1
logger = logging.getLogger(__name__)


class LinearVisualHandoffV1(BaseModel):
    """JSON envelope carrying prompt layout and only non-text embedding rows."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = LINEAR_VISUAL_HANDOFF_VERSION
    handoff_id: str
    format: Literal["linear_visual_embeds"] = LINEAR_VISUAL_HANDOFF_FORMAT
    transfer_mode: str
    decoder_model: str
    decoder_revision: str | None = None
    image_token_id: int
    hidden_size: int
    dtype: str
    prompt_token_ids: list[int]
    prompt_is_token_ids: list[bool]
    visual_embeds: TransferRequest

    @field_validator("handoff_id")
    @classmethod
    def _validate_handoff_id(cls, value: str) -> str:
        uuid.UUID(value)
        return value

    @model_validator(mode="after")
    def _validate_prompt_layout(self) -> "LinearVisualHandoffV1":
        sequence_length = len(self.prompt_token_ids)
        if sequence_length == 0:
            raise ValueError("prompt_token_ids must not be empty")
        if len(self.prompt_is_token_ids) != sequence_length:
            raise ValueError(
                "prompt_is_token_ids length must match prompt_token_ids length"
            )
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")

        visual_positions = [
            index
            for index, is_token_id in enumerate(self.prompt_is_token_ids)
            if not is_token_id
        ]
        if not visual_positions:
            raise ValueError("linear visual handoff contains no embedding positions")
        if any(
            self.prompt_token_ids[index] != self.image_token_id
            for index in visual_positions
        ):
            raise ValueError(
                "every embedding position must carry the image placeholder token"
            )
        if any(
            token_id == self.image_token_id and self.prompt_is_token_ids[index]
            for index, token_id in enumerate(self.prompt_token_ids)
        ):
            raise ValueError(
                "image placeholder tokens must be marked as embedding positions"
            )
        if self.visual_embeds.embeddings_shape != [
            len(visual_positions),
            self.hidden_size,
        ]:
            raise ValueError(
                "visual embedding shape must equal "
                f"[{len(visual_positions)}, {self.hidden_size}]"
            )
        if self.visual_embeds.embedding_dtype_str != self.dtype:
            raise ValueError("visual embedding dtype must match handoff dtype")
        return self


class HandoffReplayGuard:
    """Bounded single-use guard for handoff descriptors."""

    def __init__(self, capacity: int = 4096) -> None:
        if capacity < 1:
            raise ValueError("handoff replay capacity must be positive")
        self._capacity = capacity
        self._order: deque[str] = deque()
        self._seen: set[str] = set()

    def claim(self, handoff_id: str) -> None:
        if handoff_id in self._seen:
            raise ValueError(f"custom encoder handoff {handoff_id} was reused")
        self._seen.add(handoff_id)
        self._order.append(handoff_id)
        if len(self._order) > self._capacity:
            self._seen.remove(self._order.popleft())


async def stage_linear_visual_prompt(
    prompt: LinearVisualPrompt,
    sender: AbstractEmbeddingSender,
    *,
    transfer_mode: str,
    decoder_model: str,
    decoder_revision: str | None,
    model_config: Any,
) -> tuple[dict[str, Any], Any]:
    """Stage compact visual rows and return their JSON handoff and send future."""

    visual_embeds = prompt.visual_embeds
    if visual_embeds.dim() != 2:
        raise ValueError("LinearEmbedsAdapter must return a 2D visual tensor")
    if visual_embeds.device.type != "cpu":
        raise ValueError("custom encoder visual tensor must be CPU-resident")
    if not visual_embeds.is_contiguous():
        raise ValueError("custom encoder visual tensor must be contiguous")

    hidden_size = _hidden_size(model_config)
    if visual_embeds.shape[1] != hidden_size:
        raise ValueError(
            f"visual hidden size {visual_embeds.shape[1]} does not match "
            f"decoder hidden size {hidden_size}"
        )
    transfer_request, transfer_future = await sender.send_embeddings(
        visual_embeds,
        stage_embeddings=True,
    )
    handoff = LinearVisualHandoffV1(
        handoff_id=str(uuid.uuid4()),
        transfer_mode=transfer_mode,
        decoder_model=decoder_model,
        decoder_revision=decoder_revision,
        image_token_id=prompt.image_token_id,
        hidden_size=hidden_size,
        dtype=torch_dtype_to_string(visual_embeds.dtype),
        prompt_token_ids=prompt.prompt_token_ids,
        prompt_is_token_ids=prompt.prompt_is_token_ids,
        visual_embeds=transfer_request,
    )
    return handoff.model_dump(mode="json"), transfer_future


async def _cancel_handoff_transfer(
    receiver: AbstractEmbeddingReceiver,
    payload: Mapping[str, Any],
) -> None:
    transfer_payload = payload.get("visual_embeds")
    if not isinstance(transfer_payload, Mapping):
        return
    try:
        transfer_request = TransferRequest.model_validate(transfer_payload)
    except Exception:
        return
    try:
        await receiver.cancel_embeddings(transfer_request)
    except Exception:
        logger.warning(
            "Failed to cancel rejected custom-encoder handoff", exc_info=True
        )


async def receive_linear_visual_prompt(
    payload: Mapping[str, Any],
    receiver: AbstractEmbeddingReceiver,
    replay_guard: HandoffReplayGuard,
    *,
    expected_transfer_mode: str,
    expected_decoder_model: str,
    expected_decoder_revision: str | None,
    model_config: Any,
) -> EmbedsPrompt:
    """Validate and receive visual rows, then materialize one dense vLLM prompt."""

    try:
        handoff = LinearVisualHandoffV1.model_validate(payload)
        replay_guard.claim(handoff.handoff_id)
        expected_hidden_size = _hidden_size(model_config)
        expected_dtype = getattr(model_config, "dtype", None)
        expected_dtype_name = (
            torch_dtype_to_string(expected_dtype)
            if isinstance(expected_dtype, torch.dtype)
            else None
        )
        if handoff.transfer_mode != expected_transfer_mode:
            raise ValueError(
                f"custom encoder transfer mode {handoff.transfer_mode!r} does not "
                f"match PD mode {expected_transfer_mode!r}"
            )
        if handoff.decoder_model != expected_decoder_model:
            raise ValueError(
                f"custom encoder decoder model {handoff.decoder_model!r} does not "
                f"match PD model {expected_decoder_model!r}"
            )
        if handoff.decoder_revision != expected_decoder_revision:
            raise ValueError("custom encoder and PD decoder revisions do not match")
        if handoff.hidden_size != expected_hidden_size:
            raise ValueError(
                f"custom encoder hidden size {handoff.hidden_size} does not match "
                f"PD hidden size {expected_hidden_size}"
            )
        if expected_dtype_name is not None and handoff.dtype != expected_dtype_name:
            raise ValueError(
                f"custom encoder dtype {handoff.dtype} does not match PD dtype "
                f"{expected_dtype_name}"
            )
    except Exception:
        await _cancel_handoff_transfer(receiver, payload)
        raise

    tensor_id: int | None = None
    try:
        tensor_id, received = await receiver.receive_embeddings(handoff.visual_embeds)
        if list(received.shape) != handoff.visual_embeds.embeddings_shape:
            raise ValueError("received visual embedding shape changed during transfer")
        if torch_dtype_to_string(received.dtype) != handoff.dtype:
            raise ValueError("received visual embedding dtype changed during transfer")

        # vLLM's mixed prompt contract is dense. Allocate it only on the final
        # consumer and scatter directly from the receiver ring view before the
        # ring slot is released; no intermediate clone is needed.
        prompt_embeds = torch.zeros(
            (len(handoff.prompt_token_ids), handoff.hidden_size),
            dtype=received.dtype,
            device=received.device,
        )
        visual_mask = torch.tensor(
            [not is_token_id for is_token_id in handoff.prompt_is_token_ids],
            dtype=torch.bool,
            device=received.device,
        )
        prompt_embeds[visual_mask] = received
    except BaseException:
        if tensor_id is None:
            try:
                await receiver.cancel_embeddings(handoff.visual_embeds)
            except Exception:
                logger.warning(
                    "Failed to cancel custom-encoder handoff after receive error",
                    exc_info=True,
                )
        raise
    finally:
        if tensor_id is not None:
            receiver.release_tensor(tensor_id)

    return EmbedsPrompt(
        prompt_embeds=prompt_embeds,
        prompt_token_ids=handoff.prompt_token_ids,
        prompt_is_token_ids=handoff.prompt_is_token_ids,
    )
