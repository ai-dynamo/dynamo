# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU Memory Service server components.

Architecture:
- GMSRPCServer: Async single-threaded server with state derived from connections
- RequestHandler: Stateless business logic for allocations and metadata
- GMSServerMemoryManager: CUDA VMM physical memory allocations
- GMSMetadataStore: Key-value store for tensor metadata

Key design principle: State is DERIVED from actual connection objects, not
computed or predicted. This eliminates race conditions.

Note: The RPC client is in gpu_memory_service.client.rpc.GMSRPCClient
"""

from gpu_memory_service.common.types import ConnectionMode, ServerState, StateSnapshot
from gpu_memory_service.server.handler import RequestHandler
from gpu_memory_service.server.locking import Connection, GlobalLockFSM
from gpu_memory_service.server.memory_manager import (
    AllocationInfo,
    AllocationNotFoundError,
    GMSServerMemoryManager,
)
from gpu_memory_service.server.metadata_store import GMSMetadataStore
from gpu_memory_service.server.rpc import GMSRPCServer

__all__ = [
    "GMSRPCServer",
    "GMSServerMemoryManager",
    "AllocationInfo",
    "AllocationNotFoundError",
    "GMSMetadataStore",
    "Connection",
    "ConnectionMode",
    "RequestHandler",
    "ServerState",
    "GlobalLockFSM",
    "StateSnapshot",
]
