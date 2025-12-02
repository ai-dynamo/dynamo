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

from kvbm._core import v2 as _v2

KvbmRuntime = _v2.KvbmRuntime
ConnectorLeader = _v2.ConnectorLeader

# TODO: Re-enable when v2 connector bindings are updated
# These classes need to be updated for v2 API changes in kvbm crate
# KvbmRequest = _v2.KvbmRequest
# RustKvConnectorLeader = _v2.KvConnectorLeader
# RustSchedulerOutput = _v2.RustSchedulerOutput

# Import the handshake metadata type from worker module
from .worker import NovaPeerMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVConnectorHandshakeMetadata,
    )
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheConfig
    from vllm.v1.core.sched.output import SchedulerOutput
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
        self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig, **kwargs
    ):
        """Initialize the scheduler connector leader."""
        print("[KVBM DEBUG] SchedulerConnectorLeader.__init__ START", flush=True)

        self.vllm_config = vllm_config
        self.vllm_kv_cache_config = kv_cache_config
        self.kvbm_override_config = kwargs.get("kvbm_override_config", None)

        # JSON config has highest priority (overrides env vars and TOML files)
        self.runtime = KvbmRuntime.build_leader(self.kvbm_override_config)

        # Create leader service for coordination (separate from runtime)
        self.leader = ConnectorLeader(self.runtime)

        # Track active slots (request_id -> True)
        # TODO: SlotManager
        self._slots: dict[str, bool] = {}

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
        """
        Always returns (0, False) indicating no external tokens available.

        Returns:
            (0, False): No external tokens, no async loading
        """
        self._ensure_slot(request.request_id)
        # No external tokens available - return (0, False)
        matched_tokens = 0
        print(
            f"[KVBM] get_num_new_matched_tokens: request={request.request_id}, "
            f"computed={num_computed_tokens}, matched={matched_tokens} (cache_hits=0)"
        )
        return (matched_tokens, False)

        # TODO: Re-enable when v2 connector bindings are updated
        # total_tokens = getattr(request, "num_tokens", 0)
        # return self._connector.get_num_new_matched_tokens(
        #     request.request_id,
        #     total_tokens,
        #     num_computed_tokens,
        # )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        """
        No-op since we never have external tokens.

        This should never be called with num_external_tokens > 0.
        """
        self._ensure_slot(request.request_id)
        # No-op for Phase 1

        # TODO: Re-enable when v2 connector bindings are updated
        # block_ids = [int(block_id) for block_id in blocks.get_block_ids()[0]]
        # self._connector.update_state_after_alloc(
        #     request.request_id,
        #     block_ids,
        #     num_external_tokens,
        # )

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> bytes:
        """
        Build connector metadata for workers.

        Workers are already initialized via configure_layouts RPC in
        set_xfer_handshake_metadata, so we just return empty bytes.

        Returns:
            bytes: Empty bytes (no additional metadata needed)
        """
        return b""

        # TODO: Re-enable when v2 connector bindings are updated
        # output = RustSchedulerOutput()
        #
        # for req in scheduler_output.scheduled_new_reqs:
        #     prompt_ids = [int(token) for token in req.prompt_token_ids]
        #     block_ids = [int(block_id) for block_id in req.block_ids[0]]
        #     output.add_new_request(
        #         req.req_id,
        #         prompt_ids,
        #         block_ids,
        #         int(req.num_computed_tokens),
        #     )
        #
        # cached = scheduler_output.scheduled_cached_reqs
        # if cached is not None:
        #     for (
        #         req_id,
        #         resumed_from_preemption,
        #         new_token_ids,
        #         new_block_ids,
        #         num_computed_tokens,
        #     ) in zip(
        #         cached.req_ids,
        #         cached.resumed_from_preemption,
        #         cached.new_token_ids,
        #         cached.new_block_ids,
        #         cached.num_computed_tokens,
        #     ):
        #         tokens = (
        #             [int(token) for token in new_token_ids] if new_token_ids else []
        #         )
        #         block_ids = (
        #             [int(block_id) for block_id in new_block_ids[0]]
        #             if new_block_ids is not None and len(new_block_ids) > 0
        #             else []
        #         )
        #         output.add_cached_request(
        #             req_id,
        #             bool(resumed_from_preemption),
        #             tokens,
        #             block_ids,
        #             int(num_computed_tokens),
        #         )
        #
        # counts_source = getattr(scheduler_output, "num_scheduled_tokens", None)
        # if counts_source:
        #     counts = {
        #         str(req_id): int(value) for req_id, value in counts_source.items()
        #     }
        #     output.set_num_scheduled_tokens(counts)
        #
        # return bytes(self._connector.build_connector_metadata(output))

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Never delays block freeing.

        Returns:
            (False, None): Don't delay block freeing, no KV transfer params
        """
        # Remove slot tracking
        self._slots.pop(request.request_id, None)
        print(
            f"[KVBM] request_finished: request={request.request_id}, "
            f"blocks={len(block_ids)}, pending=False"
        )
        # Don't delay block freeing
        return (False, None)

        # TODO: Re-enable when v2 connector bindings are updated
        # assert self._connector.has_slot(
        #     request.request_id
        # ), f"Slot not found for request {request.request_id}"
        # delay = self._connector.request_finished(request.request_id)
        # return (delay, None)

    def update_connector_output(self, connector_output) -> None:
        """No-op for Phase 1."""
        pass

    def get_finished_count(self) -> Optional[int]:
        """No finished count tracking for Phase 1."""
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

    def _ensure_slot(self, request_id: str) -> None:
        """Ensure we're tracking this request."""
        if request_id not in self._slots:
            self._slots[request_id] = True

    # TODO: Re-enable when v2 connector bindings are updated
    # def _create_slot(self, request: "Request") -> None:
    #     if self._connector.has_slot(request.request_id):
    #         return
    #
    #     if bool(getattr(request, "mm_features", None)) or bool(
    #         getattr(request, "mm_positions", None)
    #     ):
    #         raise ValueError("Unsupported request - requires mm extra keys")
    #
    #     all_token_ids = [
    #         [int(token) for token in tokens] for tokens in request.all_token_ids
    #     ]
    #
    #     kv_request = KvbmRequest(
    #         request_id=request.request_id,
    #         lora_name=request.lora_request.lora_name()
    #         if request.lora_request
    #         else None,
    #         salt_hash=str(getattr(request, "cache_salt", None))
    #         if getattr(request, "cache_salt", None) is not None
    #         else None,
    #     )
    #
    #     self._connector.create_slot(kv_request, all_token_ids)
