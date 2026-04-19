# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM KV-connector leader wired to the Rust ConnectorLeader.

The Python class adapts vLLM's KVConnectorBase_V1 scheduler-side hooks to
the Rust `ConnectorLeader` in `lib/kvbm-connector/src/connector/leader/`,
which owns the InstanceLeader, the OffloadEngine (G1→G2→G3), per-request
RequestSlot state, and the cache-hit / forward-pass telemetry.

Bring-up flow:
1. `__init__` builds a `KvbmRuntime` (Velo messenger + tokio) and a
   `ConnectorLeader` over it.
2. vLLM calls `set_xfer_handshake_metadata` with each worker's
   `VeloPeerMetadata`; the leader registers them as Velo peers
   (rank-ordered) and then `initialize_workers()` drives the async
   layout-config gather and computes G2/G3 block counts from
   `cache.host.num_blocks` / `cache.host.cache_size_gb`.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Optional

import kvbm
from kvbm.v2.vllm import KvbmVllmConfig

from ..sched_output import process_scheduler_output
from .worker import VeloPeerMetadata

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
    vLLM KV-connector leader backed by the Rust ConnectorLeader.

    Responsibilities on the scheduler side:
    - Own per-request RequestSlots (created lazily in `get_num_new_matched_tokens`).
    - Drive prefix-cache lookups against the G2/G3 tiers and surface the
      matched-token count back to vLLM.
    - Build per-iteration connector metadata from the vLLM SchedulerOutput,
      including intra-pass onboard requests and the forward-pass completion
      precondition handed to workers in `bind_connector_metadata`.
    - Own Velo peer registration and trigger leader-driven worker init
      (layout gather + G2/G3 block count resolution) on `set_xfer_handshake_metadata`.
    - Track finished requests and gate block-freeing on offload completion.

    `KVBM_DECODE_OFFLOAD=true` enables opportunistic offload during decode
    by re-syncing slot tokens from the live vLLM Request each iteration.
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

        self.enable_decode_offload = os.getenv("KVBM_DECODE_OFFLOAD", "false") == "true"
        print(
            f"SchedulerConnectorLeader: enable_decode_offload: {self.enable_decode_offload}",
            flush=True,
        )

        instance_id = self.runtime.instance_id()
        print(
            f"SchedulerConnectorLeader initialized with Velo instance: {instance_id.hex()[:8]}...",
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
        Forward the post-allocation state to the Rust slot.

        vLLM hands us the device block IDs it just allocated for `request`
        and any external (matched) token count from `get_num_new_matched_tokens`.
        The Rust ConnectorLeader records the G1 destinations and — when
        `num_external_tokens > 0` — queues the corresponding G2→G1 onboard
        request that is emitted via the next connector metadata build.
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
        if self.enable_decode_offload:
            for req_id, _ in self.inflight_requests.items():
                self.update_slot(req_id)
        output = process_scheduler_output(self.iteration, scheduler_output)
        result = bytes(self.leader.build_connector_metadata(output))
        return result

    def request_finished(
        self,
        request: "Request",
        _block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Ask the Rust slot whether block freeing must be delayed.

        The Rust side returns `Pending` while an offload from the
        request's G1 blocks is still in flight — in that case we tell
        vLLM to delay block freeing so the offload can finish reading
        device memory. `Finished` / untracked requests return `False`.

        Returns:
            (delay, None): `delay=True` if the Rust slot is still
            offloading; the second element is always None (KVBM does
            not use vLLM's KV transfer params channel).
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
        Register all workers as Velo peers (rank-ordered) and drive init.

        Called by vLLM once after all TP workers have exported their
        `VeloPeerMetadata`. Each (instance_id, worker_address) pair is
        registered with the Velo messenger and wrapped in a
        `ConnectorWorkerClient` + `VeloWorkerClient` on the Rust side.
        The final `initialize_workers()` call gathers SerializedLayout
        from every worker, computes G2/G3 block counts from the runtime
        config's `cache.host` / `cache.disk` blocks, and spins up the
        OffloadEngine — failure here aborts bring-up.

        Raises:
            ValueError: if the TP ranks are not a consecutive 0..N-1 range,
                or if any entry is not a `VeloPeerMetadata`.
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
            if not isinstance(worker_meta, VeloPeerMetadata):
                raise ValueError(
                    f"Expected VeloPeerMetadata, got {type(worker_meta).__name__}"
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

        # vLLM carries connector-specific transfer params as a
        # dict[str, Any] | None on the Request. Access the attribute
        # directly so a future vLLM rename surfaces as AttributeError
        # rather than silently masking the regression. Serialize with
        # allow_nan=False so NaN / Infinity raise ValueError on this
        # side instead of producing a payload serde_json will reject
        # with a less informative error; TypeError on unserializable
        # values likewise propagates unchanged.
        kv_transfer_params = request.kv_transfer_params
        kv_transfer_params_json = (
            json.dumps(kv_transfer_params, allow_nan=False)
            if kv_transfer_params is not None
            else None
        )

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
            kv_transfer_params_json=kv_transfer_params_json,
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
