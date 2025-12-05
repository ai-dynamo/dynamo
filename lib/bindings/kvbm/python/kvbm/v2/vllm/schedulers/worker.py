# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal scheduler connector worker implementation for testing.

This is a barebones implementation that provides no-op responses,
used specifically for scheduler integration testing without actual KV transfer.

For Phase 1, the worker instantiates a KvbmRuntime with Nova and returns
peer information via get_handshake_metadata() for the leader to connect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

# Import KvbmRuntime and ConnectorWorker from Rust bindings
from kvbm._core import v2 as _v2
from kvbm.v2.vllm import KvbmVllmConfig

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
)
from vllm.model_executor.models.utils import extract_layer_index

from ..config import extract_vllm_config_for_kvbm

KvbmRuntime = _v2.KvbmRuntime
ConnectorWorker = _v2.ConnectorWorker


if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheConfig


@dataclass
class NovaPeerMetadata(KVConnectorHandshakeMetadata):
    """
    Nova peer info for handshake between worker and leader.

    This metadata is returned by get_handshake_metadata() and contains
    the serialized Nova PeerInfo needed for the leader to register this
    worker as a peer.
    """

    instance_id: bytes  # 16-byte UUID
    worker_address: bytes  # JSON-serialized WorkerAddress


class SchedulerConnectorWorker:
    """
    Minimal scheduler connector worker that provides no-op implementations.

    This connector is used for scheduler integration where no actual
    KV transfer is needed. All methods are no-ops or return minimal responses.

    In Phase 1, the worker:
    - Builds a KvbmRuntime with Nova (no etcd discovery)
    - Returns Nova peer info via get_handshake_metadata()
    """

    def __init__(
        self, vllm_config: "VllmConfig", kvbm_config: KvbmVllmConfig, kv_cache_config: KVCacheConfig, **kwargs
    ):
        """Initialize the scheduler connector worker."""
        self.vllm_config = vllm_config
        self.kvbm_config = kvbm_config
        self.vllm_kv_cache_config = kv_cache_config
        self.kvbm_override_config = kwargs.get("kvbm_override_config", None)

        # Build KvbmRuntime with Nova
        self.runtime = KvbmRuntime.build_worker(self.kvbm_override_config)

        # Create the Rust ConnectorWorker that handles NIXL registration
        self.connector_worker = ConnectorWorker(self.runtime)

        # Store peer info for handshake
        instance_id, worker_addr = self.runtime.peer_info()
        self._handshake_metadata = NovaPeerMetadata(
            instance_id=instance_id,
            worker_address=worker_addr,
        )

        # Will be set during register_kv_caches
        self._num_device_blocks: Optional[int] = None

        print(
            f"SchedulerConnectorWorker initialized with Nova instance: {instance_id.hex()[:8]}...",
            flush=True,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """
        Register KV caches with NIXL for RDMA transfers.

        This registers the KV cache tensors with NIXL via the UCX backend,
        enabling remote GPU-to-GPU transfers.
        """
        if not kv_caches:
            print("Warning: register_kv_caches called with empty kv_caches")
            return

        print(
            f"SchedulerConnectorWorker.register_kv_caches called with {len(kv_caches)} layers"
        )

        # Sort tensors by layer index to ensure correct ordering
        ordered_kv_caches = sorted(
            kv_caches.items(), key=lambda item: extract_layer_index(item[0])
        )

        # Extract tensors in order
        tensors = [tensor for _, tensor in ordered_kv_caches]

        # Get first tensor to extract common properties
        first_tensor = tensors[0]
        shape = first_tensor.shape

        # Validate all tensors have same shape
        if not all(t.shape == shape for t in tensors):
            raise NotImplementedError(
                "Hybrid models with different KV cache shapes are not supported yet."
            )

        # Extract parameters
        # For NHD layout: [2 (K/V), num_blocks, block_size, num_heads, head_size]
        # For HND layout: [2 (K/V), num_blocks, num_heads, block_size, head_size]
        num_device_blocks = max(shape[0], shape[1])
        page_size = self.kvbm_config.block_size()
        dtype_width_bytes = self.kvbm_config.cache_dtype_bytes()

        config_gpu_blocks = self.vllm_config.cache_config.num_gpu_blocks
        if num_device_blocks != config_gpu_blocks:
            print(
                f"Warning: num_device_blocks from tensor ({num_device_blocks}) "
                f"!= config.num_gpu_blocks ({config_gpu_blocks}). "
                f"Using tensor-derived value."
            )

        # Phase 2A: Register KV caches with NIXL via Rust binding
        # This caches tensor state for deferred NIXL registration
        # The actual NIXL registration happens when the leader triggers
        # initialization via bind_connector_metadata()
        self.connector_worker.register_kv_caches(
            tensors,
            num_device_blocks,
            page_size,
            dtype_width_bytes,
        )

        # Store device block count for later use
        self._num_device_blocks = num_device_blocks

        print("[KVBM] KV caches registered (deferred mode)")
        print(f"  - Num device blocks: {num_device_blocks}")
        print(f"  - Num layers: {len(tensors)}")
        print(f"  - Page size: {page_size}")
        print(f"  - Dtype width bytes: {dtype_width_bytes}")
        print(f"  - Shape: {shape}")
        print("[KVBM] Waiting for leader to trigger initialization...")

    def bind_connector_metadata(self, data: bytes) -> None:
        """
        Bind connector metadata from the leader.
        """
        self.connector_worker.bind_connector_metadata(data)

    def clear_connector_metadata(self) -> None:
        """
        Clear connector metadata - no-op.
        """
        pass

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading KV cache - no-op.

        No KV loading needed for scheduler connector.
        """
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """
        Save KV layer - no-op.

        No KV saving needed for scheduler connector.
        """
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op - no async loading."""
        pass

    def wait_for_save(self) -> None:
        """No-op - no async saving."""
        pass

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """
        Get finished request IDs.

        Since request_finished() always returns False (never delays block freeing),
        we just acknowledge the finished requests but don't return any as finished
        for KV transfer purposes.

        Returns:
            (None, None): No finished sends/receives
        """
        # Just acknowledge the finished requests
        # Since our leader's request_finished() always returns False,
        # these requests have already had their blocks freed
        if len(finished_req_ids) > 0:
            print(
                f"SchedulerConnectorWorker.get_finished() acknowledging {len(finished_req_ids)} finished requests"
            )

        return (None, None)

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Returns empty set - no load errors tracked."""
        return set()

    def get_handshake_metadata(self) -> KVConnectorHandshakeMetadata:
        """
        Return Nova peer info for leader to connect.

        Returns:
            NovaPeerMetadata containing instance_id and worker_address bytes
            that the leader will use to register this worker as a Nova peer.
        """
        return self._handshake_metadata
