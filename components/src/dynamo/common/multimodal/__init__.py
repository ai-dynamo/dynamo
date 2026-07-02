# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal utilities for Dynamo components.

The media loaders are imported eagerly (they only depend on PIL/numpy and the
local http helpers). The embedding-transfer and encoder-cache members pull in
``torch`` transitively, so they are resolved lazily via PEP 562 to keep
``from dynamo.common.multimodal import ImageLoader`` importable in CPU-only /
lightweight environments (see issue #11172).
"""

from dynamo.common.multimodal.audio_loader import AudioLoader
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.multimodal.video_loader import VideoLoader

_EMBEDDING_TRANSFER_NAMES = {
    "AbstractEmbeddingReceiver",
    "AbstractEmbeddingSender",
    "LocalEmbeddingReceiver",
    "LocalEmbeddingSender",
    "NixlReadEmbeddingReceiver",
    "NixlReadEmbeddingSender",
    "NixlWriteEmbeddingReceiver",
    "NixlWriteEmbeddingSender",
    "TransferRequest",
}

__all__ = [
    "AsyncEncoderCache",
    "AudioLoader",
    "EMBEDDING_RECEIVER_FACTORIES",
    "EMBEDDING_SENDER_FACTORIES",
    "ImageLoader",
    "VideoLoader",
    "NixlReadEmbeddingReceiver",
    "NixlReadEmbeddingSender",
    "NixlWriteEmbeddingSender",
    "NixlWriteEmbeddingReceiver",
    "TransferRequest",
    "LocalEmbeddingReceiver",
    "LocalEmbeddingSender",
]


def _build_embedding_factories() -> None:
    from collections.abc import Callable

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
    )

    sender_factories: dict[
        EmbeddingTransferMode, Callable[[], AbstractEmbeddingSender]
    ] = {
        EmbeddingTransferMode.LOCAL: LocalEmbeddingSender,
        EmbeddingTransferMode.NIXL_WRITE: NixlWriteEmbeddingSender,
        EmbeddingTransferMode.NIXL_READ: NixlReadEmbeddingSender,
    }

    receiver_factories: dict[
        EmbeddingTransferMode, Callable[[], AbstractEmbeddingReceiver]
    ] = {
        EmbeddingTransferMode.LOCAL: LocalEmbeddingReceiver,
        EmbeddingTransferMode.NIXL_WRITE: NixlWriteEmbeddingReceiver,
        # [gluo FIXME] can't use pre-registered tensor as NIXL requires descriptors
        # to be at matching size, need to overwrite nixl connect library
        EmbeddingTransferMode.NIXL_READ: lambda: NixlReadEmbeddingReceiver(max_items=0),
    }

    globals()["EMBEDDING_SENDER_FACTORIES"] = sender_factories
    globals()["EMBEDDING_RECEIVER_FACTORIES"] = receiver_factories


def __getattr__(name: str):
    if name == "AsyncEncoderCache":
        from dynamo.common.multimodal.async_encoder_cache import AsyncEncoderCache

        globals()[name] = AsyncEncoderCache
        return AsyncEncoderCache
    if name in _EMBEDDING_TRANSFER_NAMES:
        from dynamo.common.multimodal import embedding_transfer

        value = getattr(embedding_transfer, name)
        globals()[name] = value
        return value
    if name in {"EMBEDDING_SENDER_FACTORIES", "EMBEDDING_RECEIVER_FACTORIES"}:
        _build_embedding_factories()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
