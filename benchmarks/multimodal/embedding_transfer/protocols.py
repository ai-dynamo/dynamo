# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel

from dynamo.common.multimodal.embedding_transfer import TransferRequest


class TransferConfig(BaseModel):
    use_gpu: bool
    tensor_count_per_request: int
    # 'local': use local file implementation
    # 'nixl_write': use NIXL writer as initiator (direct NIXL API calls)
    # 'nixl_read': use NIXL reader as initiator (nixl_connect)
    transmitter_type: str = "local"


class TransferRequest(BaseModel):
    requests: list[TransferRequest]


class AgentRequest(BaseModel):
    agent_id: str
    agent_metadata: str
