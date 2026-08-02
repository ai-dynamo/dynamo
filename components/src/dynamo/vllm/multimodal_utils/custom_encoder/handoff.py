# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned handoff for encoder-produced linear ``EmbedsPrompt`` inputs."""

from __future__ import annotations

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
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.model_config import (
    _hidden_size,
)

LINEAR_EMBEDS_HANDOFF_FORMAT = "linear_embeds_prompt"
LINEAR_EMBEDS_HANDOFF_VERSION = 1


class LinearEmbedsHandoffV1(BaseModel):
    """JSON envelope for one encoder-produced, decoder-ready prompt tensor."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = LINEAR_EMBEDS_HANDOFF_VERSION
    handoff_id: str
    format: Literal["linear_embeds_prompt"] = LINEAR_EMBEDS_HANDOFF_FORMAT
    transfer_mode: str
    decoder_model: str
    decoder_revision: str | None = None
    image_token_id: int
    hidden_size: int
    dtype: str
    prompt_token_ids: list[int]
    prompt_is_token_ids: list[bool]
    prompt_embeds: TransferRequest

    @field_validator("handoff_id")
    @classmethod
    def _validate_handoff_id(cls, value: str) -> str:
        uuid.UUID(value)
        return value

    @model_validator(mode="after")
    def _validate_prompt_layout(self) -> "LinearEmbedsHandoffV1":
        sequence_length = len(self.prompt_token_ids)
        if sequence_length == 0:
            raise ValueError("prompt_token_ids must not be empty")
        if len(self.prompt_is_token_ids) != sequence_length:
            raise ValueError(
                "prompt_is_token_ids length must match prompt_token_ids length"
            )
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.prompt_embeds.embeddings_shape != [
            sequence_length,
            self.hidden_size,
        ]:
            raise ValueError(
                "prompt embedding shape must equal "
                f"[{sequence_length}, {self.hidden_size}]"
            )
        if self.prompt_embeds.embedding_dtype_str != self.dtype:
            raise ValueError("prompt embedding dtype must match handoff dtype")

        embedded_positions = [
            index
            for index, is_token_id in enumerate(self.prompt_is_token_ids)
            if not is_token_id
        ]
        if not embedded_positions:
            raise ValueError("linear prompt handoff contains no embedding positions")
        if any(
            self.prompt_token_ids[index] != self.image_token_id
            for index in embedded_positions
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


async def stage_linear_embeds_prompt(
    prompt: EmbedsPrompt,
    sender: AbstractEmbeddingSender,
    *,
    transfer_mode: str,
    decoder_model: str,
    decoder_revision: str | None,
    image_token_id: int,
    model_config: Any,
) -> tuple[dict[str, Any], Any]:
    """Stage a prepared prompt and return its JSON handoff plus send future."""

    prompt_embeds = prompt.get("prompt_embeds")
    prompt_token_ids = prompt.get("prompt_token_ids")
    prompt_is_token_ids = prompt.get("prompt_is_token_ids")
    if not isinstance(prompt_embeds, torch.Tensor) or prompt_embeds.dim() != 2:
        raise ValueError("LinearEmbedsAdapter must return a 2D prompt tensor")
    if prompt_embeds.device.type != "cpu":
        raise ValueError("custom encoder prompt tensor must be CPU-resident")
    if not isinstance(prompt_token_ids, list) or not isinstance(
        prompt_is_token_ids, list
    ):
        raise ValueError("LinearEmbedsAdapter must return token IDs and a token mask")

    hidden_size = _hidden_size(model_config)
    if prompt_embeds.shape[1] != hidden_size:
        raise ValueError(
            f"prompt hidden size {prompt_embeds.shape[1]} does not match "
            f"decoder hidden size {hidden_size}"
        )
    transfer_request, transfer_future = await sender.send_embeddings(
        prompt_embeds,
        stage_embeddings=True,
    )
    handoff = LinearEmbedsHandoffV1(
        handoff_id=str(uuid.uuid4()),
        transfer_mode=transfer_mode,
        decoder_model=decoder_model,
        decoder_revision=decoder_revision,
        image_token_id=image_token_id,
        hidden_size=hidden_size,
        dtype=torch_dtype_to_string(prompt_embeds.dtype),
        prompt_token_ids=prompt_token_ids,
        prompt_is_token_ids=prompt_is_token_ids,
        prompt_embeds=transfer_request,
    )
    return handoff.model_dump(mode="json"), transfer_future


async def _cancel_handoff_transfer(
    receiver: AbstractEmbeddingReceiver,
    payload: Mapping[str, Any],
) -> None:
    transfer_payload = payload.get("prompt_embeds")
    if not isinstance(transfer_payload, Mapping):
        return
    try:
        transfer_request = TransferRequest.model_validate(transfer_payload)
    except Exception:
        return
    await receiver.cancel_embeddings(transfer_request)


async def receive_linear_embeds_prompt(
    payload: Mapping[str, Any],
    receiver: AbstractEmbeddingReceiver,
    replay_guard: HandoffReplayGuard,
    *,
    expected_transfer_mode: str,
    expected_decoder_model: str,
    expected_decoder_revision: str | None,
    model_config: Any,
) -> EmbedsPrompt:
    """Validate, receive, own, and reconstruct an encoder-produced prompt."""

    try:
        handoff = LinearEmbedsHandoffV1.model_validate(payload)
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
        tensor_id, received = await receiver.receive_embeddings(handoff.prompt_embeds)
        if list(received.shape) != handoff.prompt_embeds.embeddings_shape:
            raise ValueError("received prompt embedding shape changed during transfer")
        if torch_dtype_to_string(received.dtype) != handoff.dtype:
            raise ValueError("received prompt embedding dtype changed during transfer")
        owned_prompt_embeds = received.clone().contiguous()
    except BaseException:
        if tensor_id is None:
            await receiver.cancel_embeddings(handoff.prompt_embeds)
        raise
    finally:
        if tensor_id is not None:
            receiver.release_tensor(tensor_id)

    return EmbedsPrompt(
        prompt_embeds=owned_prompt_embeds,
        prompt_token_ids=handoff.prompt_token_ids,
        prompt_is_token_ids=handoff.prompt_is_token_ids,
    )
