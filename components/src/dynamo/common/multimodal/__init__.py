# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal utilities for Dynamo components.

`AsyncEncoderCache` and the embedding-transfer members import `torch`, while the
image, audio and video loaders do not. Re-exporting all of them eagerly meant
`from dynamo.common.multimodal import ImageLoader` pulled in the whole ML stack,
so the loaders could not be imported or unit-tested without it. The heavy names
resolve on first access instead (PEP 562); the loaders stay eager because they
cost nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dynamo.common.multimodal.audio_loader import AudioLoader
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.multimodal.video_loader import VideoLoader

if TYPE_CHECKING:
    from collections.abc import Callable

    from dynamo.common.constants import EmbeddingTransferMode
    from dynamo.common.multimodal.async_encoder_cache import AsyncEncoderCache
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

    # Declared, not assigned: the runtime values come from `__getattr__` below.
    # Without these mypy widens both to `Any` and callers lose the key and value
    # types.
    EMBEDDING_SENDER_FACTORIES: dict[
        EmbeddingTransferMode, Callable[[], AbstractEmbeddingSender]
    ]
    EMBEDDING_RECEIVER_FACTORIES: dict[
        EmbeddingTransferMode, Callable[[], AbstractEmbeddingReceiver]
    ]

_EMBEDDING_TRANSFER_EXPORTS = frozenset(
    {
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
)


def _build_sender_factories() -> (
    dict[EmbeddingTransferMode, Callable[[], AbstractEmbeddingSender]]
):
    from dynamo.common.constants import EmbeddingTransferMode
    from dynamo.common.multimodal.embedding_transfer import (
        LocalEmbeddingSender,
        NixlReadEmbeddingSender,
        NixlWriteEmbeddingSender,
    )

    return {
        EmbeddingTransferMode.LOCAL: LocalEmbeddingSender,
        EmbeddingTransferMode.NIXL_WRITE: NixlWriteEmbeddingSender,
        EmbeddingTransferMode.NIXL_READ: NixlReadEmbeddingSender,
    }


def _build_receiver_factories() -> (
    dict[EmbeddingTransferMode, Callable[[], AbstractEmbeddingReceiver]]
):
    from dynamo.common.constants import EmbeddingTransferMode
    from dynamo.common.multimodal.embedding_transfer import (
        LocalEmbeddingReceiver,
        NixlReadEmbeddingReceiver,
        NixlWriteEmbeddingReceiver,
    )

    return {
        EmbeddingTransferMode.LOCAL: LocalEmbeddingReceiver,
        EmbeddingTransferMode.NIXL_WRITE: NixlWriteEmbeddingReceiver,
        # [gluo FIXME] can't use pre-registered tensor as NIXL requires descriptors
        # to be at matching size, need to overwrite nixl connect library
        EmbeddingTransferMode.NIXL_READ: lambda: NixlReadEmbeddingReceiver(max_items=0),
    }


# Hidden from the type checker on purpose. A module-level `__getattr__` tells
# mypy the package may have any attribute, which would silently accept a typo
# like `from dynamo.common.multimodal import DoesNotExistAtAll` in every module
# that imports from here. The `TYPE_CHECKING` block above already declares the
# real names, so the checker has what it needs without the widening.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> Any:
        if name == "AsyncEncoderCache":
            from dynamo.common.multimodal.async_encoder_cache import AsyncEncoderCache

            value: Any = AsyncEncoderCache
        elif name in _EMBEDDING_TRANSFER_EXPORTS:
            from dynamo.common.multimodal import embedding_transfer

            value = getattr(embedding_transfer, name)
        elif name == "EMBEDDING_SENDER_FACTORIES":
            value = _build_sender_factories()
        elif name == "EMBEDDING_RECEIVER_FACTORIES":
            value = _build_receiver_factories()
        else:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

        # Cache on the module so repeat access skips this function entirely.
        globals()[name] = value
        return value


def __dir__() -> list[str]:
    return sorted(__all__)


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
