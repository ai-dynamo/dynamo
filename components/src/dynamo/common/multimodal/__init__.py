# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multimodal utilities for Dynamo components."""

from dynamo.common.constants import EmbeddingTransferMode
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
from dynamo.common.multimodal.image_loader import ImageLoader

EMBEDDING_SENDER_FACTORIES = {
    EmbeddingTransferMode.LOCAL: LocalEmbeddingSender,
    EmbeddingTransferMode.NIXL_WRITE: NixlWriteEmbeddingSender,
    EmbeddingTransferMode.NIXL_READ: NixlReadEmbeddingSender,
}

EMBEDDING_RECEIVER_FACTORIES = {
    EmbeddingTransferMode.LOCAL: LocalEmbeddingReceiver,
    EmbeddingTransferMode.NIXL_WRITE: NixlWriteEmbeddingReceiver,
    # [gluo FIXME] can't use pre-registered tensor as NIXL requires descriptors
    # to be at matching size, need to overwrite nixl connect library
    EmbeddingTransferMode.NIXL_READ: lambda: NixlReadEmbeddingReceiver(max_items=0),
}

__all__ = [
    "AsyncEncoderCache",
    "EMBEDDING_RECEIVER_FACTORIES",
    "EMBEDDING_SENDER_FACTORIES",
    "ImageLoader",
    "NixlReadEmbeddingReceiver",
    "NixlReadEmbeddingSender",
    "NixlWriteEmbeddingSender",
    "NixlWriteEmbeddingReceiver",
    "TransferRequest",
    "LocalEmbeddingReceiver",
    "LocalEmbeddingSender",
]
