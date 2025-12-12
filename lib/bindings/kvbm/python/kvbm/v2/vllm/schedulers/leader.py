# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal scheduler connector leader implementation for testing.

This is a barebones implementation that returns minimal/no-op responses,
used specifically for scheduler integration testing without actual KV transfer.

For Phase 1, the leader:
- Builds a KvbmRuntime with Nova (no etcd discovery)
- Receives worker peer info via set_xfer_handshake_metadata()
- Registers workers as Nova peers and tracks rank→instance_id mapping
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import kvbm
from kvbm.v2.vllm import KvbmVllmConfig

from ..sched_output import process_scheduler_output
from .worker import NovaPeerMetadata

# Import v2 bindings (requires v2 feature)
if kvbm.v2.is_available():
    KvbmRuntime = kvbm.v2.KvbmRuntime
    ConnectorLeader = kvbm.v2.ConnectorLeader
    KvbmRequest = kvbm.v2.KvbmRequest
else:
    KvbmRuntime = None
    ConnectorLeader = None
    KvbmRequest = None

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorHandshakeMetadata,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


class SchedulerConnectorLeader:
    """
    Minimal scheduler connector leader that returns no-op responses.

    This connector is used for scheduler integration where no actual
    KV transfer is needed. All methods return minimal valid responses.

    In Phase 1, the leader:
    - Builds a KvbmRuntime with Nova (no etcd discovery)
    - Receives worker peer info via set_xfer_handshake_metadata()
    - Registers workers as Nova peers and tracks rank→instance_id mapping
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kvbm_config: KvbmVllmConfig,
        kv_cache_config: KVCacheConfig,
        **kwargs,
    ):
        """Initialize the scheduler connector leader."""
        # Check that v2 feature is available
        if not kvbm.v2.is_available():
            raise ImportError(
                "SchedulerConnectorLeader requires the 'v2' feature. "
                "Rebuild kvbm with: maturin develop --features v2"
            )

        print("[KVBM DEBUG] SchedulerConnectorLeader.__init__ START", flush=True)

        self.vllm_config = vllm_config
        self.kvbm_config = kvbm_config
        self.vllm_kv_cache_config = kv_cache_config
        self.kvbm_override_config = kwargs.get("kvbm_override_config", None)
        self.inflight_requests = {}

        self.iteration = 0
        self.block_size = vllm_config.cache_config.block_size

        # JSON config has highest priority (overrides env vars and TOML files)
        self.runtime = KvbmRuntime.build_leader(self.kvbm_override_config)

        # Create leader service for coordination (separate from runtime)
        self.leader = ConnectorLeader(self.runtime, self.block_size)

        instance_id = self.runtime.instance_id()
        print(
            f"SchedulerConnectorLeader initialized with Nova instance: {instance_id.hex()[:8]}...",
            flush=True,
        )

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        self._create_slot(request)
        return self.leader.get_num_new_matched_tokens(
            request.request_id, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        """
        No-op since we never have external tokens.

        This should never be called with num_external_tokens > 0.
        """
        block_ids = [int(block_id) for block_id in blocks.get_block_ids()[0]]
        self.leader.update_state_after_alloc(
            request.request_id, block_ids, num_external_tokens
        )

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> bytes:
        """
        Build connector metadata for workers.

        This processes the vLLM scheduler output and generates connector metadata
        that workers use to execute KV transfers.

        Args:
            scheduler_output: vLLM's SchedulerOutput object

        Returns:
            bytes: Serialized connector metadata
        """
        self.iteration = self.iteration + 1
        output = process_scheduler_output(self.iteration, scheduler_output)
        return bytes(self.leader.build_connector_metadata(output))

    def request_finished(
        self,
        request: "Request",
        _block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Never delays block freeing.

        Returns:
            (False, None): Don't delay block freeing, no KV transfer params
        """
        # we only use this to update the total tokens in the slot
        # its safe to remove it if it exists
        if request.request_id in self.inflight_requests:
            del self.inflight_requests[request.request_id]
        delay = self.leader.request_finished(request.request_id)
        return (delay, None)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        # Convert None to empty sets for Rust binding compatibility
        finished_sending = (
            connector_output.finished_sending
            if connector_output.finished_sending is not None
            else set()
        )
        finished_recving = (
            connector_output.finished_recving
            if connector_output.finished_recving is not None
            else set()
        )
        self.leader.update_connector_output(finished_sending, finished_recving)

    def get_finished_count(self) -> Optional[int]:
        return None

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, "KVConnectorHandshakeMetadata"]
    ) -> None:
        """
        Register all worker Nova peers and trigger layout initialization.

        This is called by vLLM after aggregating handshake metadata from all
        TP workers. We:
        1. Register each worker as a Nova peer
        2. Track the mapping from TP rank to instance_id
        3. Determine G2/G3 layout configuration from vLLM config
        4. Send configure_layouts RPC to each worker to trigger initialization

        Args:
            metadata: Dictionary mapping tp_rank (int) to NovaPeerMetadata
        """
        # Create sorted list of (tp_rank, worker_meta) tuples sorted by tp_rank
        sorted_workers = sorted(metadata.items(), key=lambda x: x[0])

        # Validate that we have consecutive tp_ranks from 0 to N-1
        num_workers = len(sorted_workers)
        expected_ranks = list(range(num_workers))
        actual_ranks = [tp_rank for tp_rank, _ in sorted_workers]

        if actual_ranks != expected_ranks:
            raise ValueError(
                f"Expected consecutive tp_ranks from 0 to {num_workers - 1}, "
                f"got {actual_ranks}"
            )

        # Validate all metadata types and register workers in sorted order
        for tp_rank, worker_meta in sorted_workers:
            if not isinstance(worker_meta, NovaPeerMetadata):
                raise ValueError(
                    f"Expected NovaPeerMetadata, got {type(worker_meta).__name__}"
                )
            self.leader.register_worker(
                tp_rank, worker_meta.instance_id, worker_meta.worker_address
            )

        # Single call to initialize all workers
        self.leader.initialize_workers()

    # Utility functions

    # note: creates a request slot for tracking state
    def _create_slot(self, request: "Request") -> None:
        if request.request_id not in self.inflight_requests:
            self.inflight_requests[request.request_id] = request

        if self.leader.has_slot(request.request_id):
            self.update_slot(request.request_id)
            return

        if bool(getattr(request, "mm_features", None)) or bool(
            getattr(request, "mm_positions", None)
        ):
            raise ValueError("Unsupported request - requires mm extra keys")

        # For v1 API, all_token_ids is already a flat list for single-sequence
        # For multi-sequence (hybrid), it would be a list of sequences - handle both
        if isinstance(request.all_token_ids[0], (list, tuple)):
            # Multi-sequence case: take first sequence
            all_token_ids = [int(token) for token in request.all_token_ids[0]]
        else:
            # Single-sequence case: already flat
            all_token_ids = [int(token) for token in request.all_token_ids]

        kv_request = KvbmRequest(
            request_id=request.request_id,
            tokens=all_token_ids,
            lora_name=request.lora_request.lora_name()
            if request.lora_request
            else None,
            salt_hash=str(getattr(request, "cache_salt", None))
            if getattr(request, "cache_salt", None) is not None
            else None,
            max_tokens=request.max_tokens,
        )

        self.leader.create_slot(kv_request)

        # Store the vLLM Request object for later token synchronization
        self.inflight_requests[request.request_id] = request

    def update_slot(self, request_id: str) -> None:
        """
        Synchronize new tokens from the vLLM Request to the slot.

        This is called during decoding to detect when new tokens have been
        generated and extend the slot's token sequence accordingly.

        Only single-sequence (non-hybrid) requests are supported.

        This is a *HACK* because vLLM does not provide us with updated token_ids
        during generation. This method allows us to update our token sequence to
        handle eviction/restarts and new tokens being generated.

        Args:
            request_id: The request ID to update
        """
        request = self.inflight_requests.get(request_id)
        if request is None:
            return  # Request not tracked

        # Only support single-sequence (non-hybrid) case
        if isinstance(request.all_token_ids[0], (list, tuple)):
            return  # Hybrid not supported, skip update

        # if the slot doesn't exist, we can't update it
        if not self.leader.has_slot(request.request_id):
            return

        slot_token_count = self.leader.get_slot_total_tokens(request_id)
        request_token_count = len(request.all_token_ids)

        if slot_token_count < request_token_count:
            print(
                f"Updating slot {request_id} with {request_token_count - slot_token_count} new tokens",
                flush=True,
            )
            new_tokens = [int(t) for t in request.all_token_ids[slot_token_count:]]
            self.leader.extend_slot_tokens(request_id, new_tokens)
