# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal utilities for Dynamo components."""

from collections.abc import Callable
from importlib import import_module
from typing import Any

from dynamo.common.constants import EmbeddingTransferMode


_LAZY_EXPORTS = {
    "AsyncEncoderCache": "dynamo.common.multimodal.async_encoder_cache",
    "AudioLoader": "dynamo.common.multimodal.audio_loader",
    "ImageLoader": "dynamo.common.multimodal.image_loader",
    "VideoLoader": "dynamo.common.multimodal.video_loader",
    "AbstractEmbeddingReceiver": "dynamo.common.multimodal.embedding_transfer",
    "AbstractEmbeddingSender": "dynamo.common.multimodal.embedding_transfer",
    "LocalEmbeddingReceiver": "dynamo.common.multimodal.embedding_transfer",
    "LocalEmbeddingSender": "dynamo.common.multimodal.embedding_transfer",
    "NixlReadEmbeddingReceiver": "dynamo.common.multimodal.embedding_transfer",
    "NixlReadEmbeddingSender": "dynamo.common.multimodal.embedding_transfer",
    "NixlWriteEmbeddingReceiver": "dynamo.common.multimodal.embedding_transfer",
    "NixlWriteEmbeddingSender": "dynamo.common.multimodal.embedding_transfer",
    "TransferRequest": "dynamo.common.multimodal.embedding_transfer",
}


def _load_export(name: str) -> Any:
    module = import_module(_LAZY_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        return _load_export(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _lazy_factory(export_name: str, *args: Any, **kwargs: Any) -> Any:
    return _load_export(export_name)(*args, **kwargs)


def _embedding_sender_factory(export_name: str) -> Callable[[], Any]:
    return lambda: _lazy_factory(export_name)


def _embedding_receiver_factory(export_name: str) -> Callable[[], Any]:
    return lambda: _lazy_factory(export_name)


EMBEDDING_SENDER_FACTORIES: dict[EmbeddingTransferMode, Callable[[], Any]] = {
    EmbeddingTransferMode.LOCAL: _embedding_sender_factory("LocalEmbeddingSender"),
    EmbeddingTransferMode.NIXL_WRITE: _embedding_sender_factory(
        "NixlWriteEmbeddingSender"
    ),
    EmbeddingTransferMode.NIXL_READ: _embedding_sender_factory("NixlReadEmbeddingSender"),
}

EMBEDDING_RECEIVER_FACTORIES: dict[EmbeddingTransferMode, Callable[[], Any]] = {
    EmbeddingTransferMode.LOCAL: _embedding_receiver_factory("LocalEmbeddingReceiver"),
    EmbeddingTransferMode.NIXL_WRITE: _embedding_receiver_factory(
        "NixlWriteEmbeddingReceiver"
    ),
    # [gluo FIXME] can't use pre-registered tensor as NIXL requires descriptors
    # to be at matching size, need to overwrite nixl connect library
    EmbeddingTransferMode.NIXL_READ: lambda: _lazy_factory(
        "NixlReadEmbeddingReceiver", max_items=0
    ),
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
