# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal utilities for Dynamo components.

The media loaders (``ImageLoader``, ``AudioLoader``, ``VideoLoader``) are
imported eagerly: their footprint is PIL/numpy plus local HTTP helpers.

Everything else is resolved through a PEP 562 module-level ``__getattr__``
so that importing a media loader does not transitively pull in ``torch``.
``AsyncEncoderCache`` reaches ``torch`` via
``dynamo.common.memory.multimodal_embedding_cache_manager``, and
``embedding_transfer`` imports ``torch`` and ``safetensors.torch``
directly. Deferring them keeps ``from dynamo.common.multimodal import
ImageLoader`` cheap while leaving the public API unchanged --- every name in
``__all__`` still resolves on first attribute access and is then cached into
``globals()``, so repeat access is free and object identity is stable.
"""

from typing import TYPE_CHECKING, Any

from dynamo.common.multimodal.audio_loader import AudioLoader
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.multimodal.video_loader import VideoLoader

# Declared for static analysis only; resolved at runtime by ``__getattr__``
# below. Limited to the names in ``__all__`` --- the abstract bases are
# reachable at runtime too, but re-declaring them here trips ruff's F401.
if TYPE_CHECKING:
    from dynamo.common.multimodal.async_encoder_cache import AsyncEncoderCache
    from dynamo.common.multimodal.embedding_transfer import (
        LocalEmbeddingReceiver,
        LocalEmbeddingSender,
        NixlReadEmbeddingReceiver,
        NixlReadEmbeddingSender,
        NixlWriteEmbeddingReceiver,
        NixlWriteEmbeddingSender,
        TransferRequest,
    )

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

# Names re-exported from ``embedding_transfer``, resolved on first access.
_EMBEDDING_TRANSFER_NAMES = frozenset(
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

_FACTORY_NAMES = frozenset(
    {"EMBEDDING_RECEIVER_FACTORIES", "EMBEDDING_SENDER_FACTORIES"}
)


def _build_embedding_factories() -> None:
    """Build both factory dicts and install them into ``globals()``.

    The dicts are built from torch-dependent classes, so their
    *construction* --- not merely their lookup --- has to be deferred.
    Installing into ``globals()`` keeps their object identity stable across
    repeat access, which callers holding a reference rely on.
    """
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


def __getattr__(name: str) -> Any:
    if name == "AsyncEncoderCache":
        from dynamo.common.multimodal.async_encoder_cache import AsyncEncoderCache

        globals()[name] = AsyncEncoderCache
        return AsyncEncoderCache

    if name in _EMBEDDING_TRANSFER_NAMES:
        from dynamo.common.multimodal import embedding_transfer

        value = getattr(embedding_transfer, name)
        globals()[name] = value
        return value

    if name in _FACTORY_NAMES:
        _build_embedding_factories()
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
