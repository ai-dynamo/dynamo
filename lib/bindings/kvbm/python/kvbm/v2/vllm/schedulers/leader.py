# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
vLLM KV-connector leader wired to the Rust ConnectorLeader.

The Python class adapts vLLM's KVConnectorBase_V1 scheduler-side hooks to
the Rust `ConnectorLeader` in `lib/kvbm-connector/src/connector/leader/`,
which owns the InstanceLeader, the OffloadEngine (G1→G2→G3), per-request
RequestSlot state, and the cache-hit / forward-pass telemetry.

Bring-up flow:
1. `__init__` builds a `KvbmRuntime` (Velo messenger + tokio) and calls
   `_init_connector(runtime, block_size)` from the base class.
2. vLLM calls `set_xfer_handshake_metadata` with each worker's
   `VeloPeerMetadata`; the base class registers them as Velo peers
   (rank-ordered) and then calls `initialize_workers()` to drive the async
   layout-config gather and compute G2/G3 block counts from the config.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import kvbm
from kvbm.v2.connector.base import KvbmConnectorLeaderBase
from kvbm.v2.vllm import KvbmVllmConfig

from ..sched_output import process_scheduler_output

if kvbm.v2.is_available():
    KvbmRuntime = kvbm.v2.KvbmRuntime
    KvbmRequest = kvbm.v2.KvbmRequest
else:
    KvbmRuntime = None
    KvbmRequest = None

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput
    from vllm.v1.request import Request


class SchedulerConnectorLeader(KvbmConnectorLeaderBase):
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
        vllm_config: "VllmConfig",
        kvbm_config: KvbmVllmConfig,
        kv_cache_config: "KVCacheConfig",
        **kwargs,
    ):
        super().__init__()

        if not kvbm.v2.is_available():
            raise ImportError(
                "SchedulerConnectorLeader requires the 'v2' feature. "
                "Rebuild kvbm with: maturin develop --features v2"
            )

        self.vllm_config = vllm_config
        self.kvbm_config = kvbm_config
        self.vllm_kv_cache_config = kv_cache_config
        self.kvbm_override_config = kwargs.get("kvbm_override_config", None)

        block_size = vllm_config.cache_config.block_size
        runtime = KvbmRuntime.build_leader(self.kvbm_override_config)
        self._init_connector(runtime, block_size)

        instance_id = self._runtime.instance_id()
        print(
            f"SchedulerConnectorLeader initialized with Velo instance: {instance_id.hex()[:8]}...",
            flush=True,
        )
        print(
            f"SchedulerConnectorLeader: enable_decode_offload: {self.enable_decode_offload}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Abstract implementations
    # ------------------------------------------------------------------

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> bytes:
        """
        Build connector metadata for workers.

        Processes the vLLM scheduler output and generates connector metadata
        that workers use to execute KV transfers.
        """
        self._iteration += 1
        if self.enable_decode_offload:
            for req_id, request in self._inflight_requests.items():
                tokens = self._get_request_tokens(request)
                if tokens is not None:
                    self._sync_slot_tokens(req_id, tokens)
        output = process_scheduler_output(self._iteration, scheduler_output)
        return bytes(self._connector.build_connector_metadata(output))

    def _create_slot(self, request: "Request") -> None:
        """Create a RequestSlot in the Rust leader for *request*."""
        if self._connector.has_slot(request.request_id):
            tokens = self._get_request_tokens(request)
            if tokens is not None:
                self._sync_slot_tokens(request.request_id, tokens)
            return

        if bool(getattr(request, "mm_features", None)) or bool(
            getattr(request, "mm_positions", None)
        ):
            raise ValueError("Unsupported request - requires mm extra keys")

        all_token_ids = self._get_request_tokens(request)
        if all_token_ids is None:
            raise ValueError(f"Cannot extract tokens from request {request.request_id}")

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
        self._connector.create_slot(kv_request)
        self._inflight_requests[request.request_id] = request

    # ------------------------------------------------------------------
    # vLLM adapter methods
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
        self._create_slot(request)
        return self._connector.get_num_new_matched_tokens(
            request.request_id, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ) -> None:
        """
        Forward the post-allocation state to the Rust slot.

        vLLM hands us the device block IDs it just allocated for `request`
        and any external (matched) token count from `get_num_new_matched_tokens`.
        """
        block_ids = [int(block_id) for block_id in blocks.get_block_ids()[0]]
        self._connector.update_state_after_alloc(
            request.request_id, block_ids, num_external_tokens
        )

    def request_finished(
        self,
        request: "Request",
        _block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Ask the Rust slot whether block freeing must be delayed.

        Returns:
            (delay, None): `delay=True` if the Rust slot is still offloading.
        """
        delay = super().request_finished(request.request_id)
        return (delay, None)

    def update_connector_output(self, connector_output: "KVConnectorOutput") -> None:
        finished_sending = connector_output.finished_sending or set()
        finished_recving = connector_output.finished_recving or set()
        self._connector.update_connector_output(finished_sending, finished_recving)

    def get_finished_count(self) -> Optional[int]:
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_request_tokens(request: "Request") -> Optional[list[int]]:
        """Extract the flat token list from a vLLM Request."""
        if isinstance(request.all_token_ids[0], (list, tuple)):
            # Multi-sequence (hybrid): take first sequence
            return [int(t) for t in request.all_token_ids[0]]
        return [int(t) for t in request.all_token_ids]
