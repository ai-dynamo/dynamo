# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transfer CustomEncoder artifacts between Dynamo worker processes."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from dynamo.common.constants import EmbeddingTransferMode
from dynamo.common.multimodal.embedding_transfer import (
    AbstractEmbeddingReceiver,
    AbstractEmbeddingSender,
    LocalEmbeddingReceiver,
    LocalEmbeddingSender,
    NixlReadEmbeddingReceiver,
    NixlReadEmbeddingSender,
    NixlWriteEmbeddingReceiver,
    NixlWriteEmbeddingSender,
    TransferRequest,
)

CUSTOM_ENCODER_ARTIFACTS_KEY = "custom_encoder_artifacts"
_TENSOR_KIND = "tensor"


def create_custom_encoder_artifact_sender(
    mode: EmbeddingTransferMode,
) -> AbstractEmbeddingSender:
    """Create the sender paired with a remote CustomEncoder artifact receiver."""

    if mode is EmbeddingTransferMode.LOCAL:
        return LocalEmbeddingSender()
    if mode is EmbeddingTransferMode.NIXL_WRITE:
        return NixlWriteEmbeddingSender()
    if mode is EmbeddingTransferMode.NIXL_READ:
        return NixlReadEmbeddingSender(
            enable_progress_thread=False, completion_poll_ms=1
        )
    raise ValueError(f"Invalid embedding transfer mode: {mode}")


def create_custom_encoder_artifact_receiver(
    mode: EmbeddingTransferMode,
) -> AbstractEmbeddingReceiver:
    """Create the receiver paired with a CustomEncoder artifact sender."""

    if mode is EmbeddingTransferMode.LOCAL:
        return LocalEmbeddingReceiver()
    if mode is EmbeddingTransferMode.NIXL_WRITE:
        return NixlWriteEmbeddingReceiver()
    if mode is EmbeddingTransferMode.NIXL_READ:
        # CustomEncoder artifacts have dynamic row counts. Allocate exact-size
        # descriptors rather than retaining a model-specific buffer ladder.
        return NixlReadEmbeddingReceiver(
            max_items=0,
            enable_progress_thread=False,
            completion_poll_ms=1,
        )
    raise ValueError(f"Invalid embedding transfer mode: {mode}")


def _artifact_payload(artifact: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(artifact, torch.Tensor):
        return artifact, {"kind": _TENSOR_KIND}
    raise TypeError(
        "Transferred CustomEncoder artifacts must be torch.Tensor; "
        f"got {type(artifact).__name__}"
    )


async def export_custom_encoder_artifacts(
    sender: AbstractEmbeddingSender,
    artifacts: Sequence[Any],
) -> tuple[dict[str, Any], list[Awaitable[None]]]:
    """Export artifacts and return a JSON wire object plus completion handles."""

    if not artifacts:
        raise ValueError("CustomEncoder returned no artifacts to transfer")

    tensors_and_metadata = [_artifact_payload(artifact) for artifact in artifacts]
    transfers = await asyncio.gather(
        *(
            sender.send_embeddings(tensor, stage_embeddings=True)
            for tensor, _ in tensors_and_metadata
        )
    )
    entries = []
    completions: list[Awaitable[None]] = []
    for (_, metadata), (request, completion) in zip(
        tensors_and_metadata, transfers, strict=True
    ):
        entries.append({**metadata, "transfer": request.model_dump()})
        completions.append(completion)
    return {CUSTOM_ENCODER_ARTIFACTS_KEY: entries}, completions


def has_transferred_custom_encoder_artifacts(request: dict[str, Any]) -> bool:
    """Return whether a request carries the CustomEncoder transfer envelope."""

    encoder_result = request.get("encoder_result")
    return isinstance(encoder_result, dict) and (
        CUSTOM_ENCODER_ARTIFACTS_KEY in encoder_result
    )


@dataclass
class ReceivedCustomEncoderArtifacts:
    """Received artifacts and the buffers that back them."""

    artifacts: list[Any]
    _receiver: AbstractEmbeddingReceiver
    _tensor_ids: list[int]

    def release(self) -> None:
        """Release transfer buffers after the decoder adapter copies their data."""

        for tensor_id in self._tensor_ids:
            self._receiver.release_tensor(tensor_id)
        self._tensor_ids.clear()


async def receive_custom_encoder_artifacts(
    receiver: AbstractEmbeddingReceiver,
    encoder_result: Any,
) -> ReceivedCustomEncoderArtifacts:
    """Receive and reconstruct CustomEncoder artifacts from a wire envelope."""

    if not isinstance(encoder_result, dict):
        raise TypeError("encoder_result must be an object")
    entries = encoder_result.get(CUSTOM_ENCODER_ARTIFACTS_KEY)
    if not isinstance(entries, list) or not entries:
        raise ValueError(
            f"encoder_result.{CUSTOM_ENCODER_ARTIFACTS_KEY} must be a non-empty list"
        )

    requests: list[TransferRequest] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError(f"CustomEncoder transfer entry {index} must be an object")
        if entry.get("kind") != _TENSOR_KIND:
            raise ValueError(
                f"CustomEncoder transfer entry {index} has unsupported kind "
                f"{entry.get('kind')!r}"
            )
        try:
            requests.append(TransferRequest.model_validate(entry["transfer"]))
        except KeyError as exc:
            raise ValueError(
                f"CustomEncoder transfer entry {index} has no transfer descriptor"
            ) from exc

    received = await asyncio.gather(
        *(receiver.receive_embeddings(request) for request in requests),
        return_exceptions=True,
    )
    first_error = next(
        (result for result in received if isinstance(result, BaseException)), None
    )
    if first_error is not None:
        for result in received:
            if isinstance(result, tuple):
                receiver.release_tensor(result[0])
        raise first_error

    tensor_ids: list[int] = []
    artifacts: list[Any] = []
    for index, result in enumerate(received):
        if not isinstance(result, tuple):
            raise RuntimeError(
                f"CustomEncoder transfer {index} did not return a tensor"
            )
        tensor_id, tensor = result
        tensor_ids.append(tensor_id)
        artifacts.append(tensor)

    return ReceivedCustomEncoderArtifacts(artifacts, receiver, tensor_ids)
