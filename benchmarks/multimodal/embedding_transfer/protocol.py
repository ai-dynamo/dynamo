# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from dynamo.common.constants import EmbeddingTransferMode
from dynamo.common.multimodal.embedding_transfer import TransferRequest


class TransferConfig(BaseModel):
    use_gpu: bool = False
    tensor_count_per_request: int = 30
    # EmbeddingTransferMode.LOCAL: use local file implementation
    # EmbeddingTransferMode.NIXL_WRITE: use NIXL writer as initiator (direct NIXL API calls)
    # EmbeddingTransferMode.NIXL_READ: use NIXL reader as initiator (nixl_connect)
    transfer_type: EmbeddingTransferMode = EmbeddingTransferMode.LOCAL


class BatchTransferRequest(BaseModel):
    requests: list[TransferRequest]
