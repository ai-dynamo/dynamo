# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common handshake utilities for v2 connectors.

This module provides shared classes and functions for Nova peer handshake
between workers and leaders, used by both vLLM and TRTLLM connectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class NovaPeerMetadata:
    """
    Nova peer info for handshake between worker and leader.

    This metadata is returned by workers and contains the serialized
    Nova PeerInfo needed for the leader to register workers as peers.

    Attributes:
        instance_id: 16-byte UUID identifying the worker's Nova instance
        worker_address: JSON-serialized WorkerAddress for TCP connection
    """

    instance_id: bytes  # 16-byte UUID
    worker_address: bytes  # JSON-serialized WorkerAddress


@runtime_checkable
class ConnectorLeaderProtocol(Protocol):
    """Protocol for leader objects that can register workers."""

    def register_worker(
        self, rank: int, instance_id: bytes, worker_address: bytes
    ) -> None:
        """Register a worker peer with the leader."""
        ...

    def initialize_workers(self) -> None:
        """Initialize all registered workers."""
        ...


def register_workers_from_handshake(
    leader: ConnectorLeaderProtocol,
    metadata: dict[int, Any],
) -> None:
    """
    Register workers from handshake metadata.

    Shared logic for both vLLM and TRTLLM connectors. This function:
    1. Sorts workers by rank
    2. Validates consecutive ranks from 0 to N-1
    3. Type-checks metadata as NovaPeerMetadata
    4. Registers each worker with the leader
    5. Initializes all workers

    Args:
        leader: The connector leader instance (must implement ConnectorLeaderProtocol)
        metadata: Dictionary mapping rank (int) to NovaPeerMetadata

    Raises:
        ValueError: If ranks are not consecutive from 0 to N-1
        ValueError: If metadata is not NovaPeerMetadata instances
    """
    sorted_workers = sorted(metadata.items(), key=lambda x: x[0])

    num_workers = len(sorted_workers)
    expected_ranks = list(range(num_workers))
    actual_ranks = [rank for rank, _ in sorted_workers]

    if actual_ranks != expected_ranks:
        raise ValueError(
            f"Expected consecutive ranks from 0 to {num_workers - 1}, "
            f"got {actual_ranks}"
        )

    for rank, worker_meta in sorted_workers:
        if not isinstance(worker_meta, NovaPeerMetadata):
            raise ValueError(
                f"Expected NovaPeerMetadata, got {type(worker_meta).__name__}"
            )
        leader.register_worker(
            rank, worker_meta.instance_id, worker_meta.worker_address
        )

    leader.initialize_workers()
