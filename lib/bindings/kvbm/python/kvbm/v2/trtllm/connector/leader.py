# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TRTLLM v2 Connector Leader implementation.

This connector uses the v2 runtime with Nova for distributed communication,
replacing the v1 pattern of KvbmLeader + DistributedRuntime.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, List, Optional

import kvbm
from kvbm.v2.common import register_workers_from_handshake

# Import v2 bindings (requires v2 feature)
if kvbm.v2.is_available():
    KvbmRuntime = kvbm.v2.KvbmRuntime
    ConnectorLeader = kvbm.v2.ConnectorLeader
    KvbmRequest = kvbm.v2.KvbmRequest
    KvbmSchedulerOutput = kvbm.v2.SchedulerOutput
else:
    KvbmRuntime = None
    ConnectorLeader = None
    KvbmRequest = None
    KvbmSchedulerOutput = None

if TYPE_CHECKING:
    from tensorrt_llm._torch.pyexecutor.kv_cache_connector import SchedulerOutput
    from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# Conditional import - only fail when class is instantiated, not at module load
try:
    from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
        KvCacheConnectorScheduler,
    )

    _TRTLLM_AVAILABLE = True
except ImportError:
    KvCacheConnectorScheduler = object  # Stub base class
    _TRTLLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class DynamoKVBMConnectorLeader(KvCacheConnectorScheduler):
    """
    TRTLLM v2 Connector Leader using KvbmRuntime + Nova.

    This replaces the v1 pattern of KvbmLeader + DistributedRuntime with
    the v2 KvbmRuntime which uses Nova for distributed communication.
    """

    def __init__(
        self,
        llm_args: "TorchLlmArgs",
        kvbm_override_config: Optional[str] = None,
    ):
        # Check dependencies before super().__init__
        if not _TRTLLM_AVAILABLE:
            raise ImportError(
                "DynamoKVBMConnectorLeaderV2 requires tensorrt_llm. "
                "Install tensorrt_llm to use this connector."
            )

        super().__init__(llm_args)

        # Check that v2 feature is available
        if not kvbm.v2.is_available():
            raise ImportError(
                "DynamoKVBMConnectorLeaderV2 requires the 'v2' feature. "
                "Rebuild kvbm with: maturin develop --features v2"
            )

        config = {
            "leader": {
                "cache": {"host": {"cache_size_gb": 10.0}},
                "tokio": {"worker_threads": 2},
                "onboard": {"mode": "inter"},
            },
            "worker": {
                "nixl": {"backends": {"UCX": {}}},
                "tokio": {"worker_threads": 2},
            },
        }

        self.kvbm_override_config = config
        self.inflight_requests = {}
        self.iteration = 0

        mappings = self._llm_args.parallel_config.to_mapping()
        self.block_size = self._llm_args.kv_cache_config.tokens_per_block
        self.rank = mappings.rank

        # Build KvbmRuntime with Nova (replaces KvbmLeader + DRT)
        self.runtime = KvbmRuntime.build_leader(json.dumps(config))

        # Create leader service for coordination
        self.leader = ConnectorLeader(self.runtime, self.block_size)

        # Decode offload support
        self.enable_decode_offload = (
            os.getenv("KVBM_DECODE_OFFLOAD", "false").lower() == "true"
        )

        instance_id = self.runtime.instance_id()
        logger.info(
            f"DynamoKVBMConnectorLeader initialized with Nova instance: {instance_id.hex()[:8]}..."
        )
        logger.info(f"  rank: {self.rank}, block_size: {self.block_size}")
        logger.info(f"  enable_decode_offload: {self.enable_decode_offload}")

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> bytes:
        """
        Build the metadata for the worker.

        Args:
            scheduler_output: The data for all inflight requests.

        Returns:
            The metadata for the workers.
        """
        self.iteration += 1

        # Update slots for decode offload if enabled
        if self.enable_decode_offload:
            for req_id in self.inflight_requests:
                self._update_slot(req_id)

        # Convert TRTLLM SchedulerOutput to KVBM format
        output = KvbmSchedulerOutput(self.iteration)

        # Build num_scheduled_tokens map for all requests
        num_scheduled_tokens = {}

        for req in scheduler_output.new_requests:
            # Use all_tokens and all_block_ids for full prompt data (matches vLLM behavior)
            output.add_new_request(
                str(req.request_id),
                prompt_token_ids=list(req.all_tokens),
                block_ids=list(req.all_block_ids),
                num_computed_tokens=int(req.computed_position),
            )
            num_scheduled_tokens[str(req.request_id)] = req.num_scheduled_tokens

        for req in scheduler_output.cached_requests:
            output.add_cached_request(
                str(req.request_id),
                resumed=False,  # TRTLLM doesn't have preemption tracking yet
                new_token_ids=list(req.new_tokens),
                all_token_ids=None,
                new_block_ids=list(req.new_block_ids),
                num_computed_tokens=int(req.computed_position),
                num_output_tokens=0,
            )
            num_scheduled_tokens[str(req.request_id)] = req.num_scheduled_tokens

        # Set num_scheduled_tokens - required by Rust connector to process offload
        output.set_num_scheduled_tokens(num_scheduled_tokens)

        return bytes(self.leader.build_connector_metadata(output))

    def get_num_new_matched_tokens(
        self, request: "LlmRequest", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        Get the number of tokens that can be loaded from remote KV cache.

        Args:
            request: The request to get the number of tokens for.
            num_computed_tokens: The number of tokens already matched on device.

        Returns:
            The number of tokens that can be loaded from remote KV cache.
            Whether the tokens will be loaded asynchronously.
        """
        self._create_slot(request)

        print(
            f"Getting num new matched tokens for request: {request.request_id}; num_computed_tokens: {num_computed_tokens}"
        )

        result = self.leader.get_num_new_matched_tokens(
            str(request.request_id), num_computed_tokens
        )

        # v2 returns (Option<int>, bool) - convert None to 0 for TRTLLM compatibility
        num_tokens, is_async = result
        # , feature_version_mismatch("async_search")
        assert num_tokens is not None
        return (num_tokens if num_tokens is not None else 0, is_async)

    def update_state_after_alloc(
        self, request: "LlmRequest", block_ids: List[int], num_external_tokens: int
    ) -> None:
        """
        Called after get_num_new_matched_tokens to provide the block ids.

        Args:
            request: The request that was allocated resources.
            block_ids: The KV cache block IDs that were allocated.
        """
        self.leader.update_state_after_alloc(
            str(request.request_id), block_ids, num_external_tokens
        )

    def request_finished(
        self, request: "LlmRequest", cache_block_ids: List[int]
    ) -> bool:
        """
        Called when a request is finished generating tokens.

        Args:
            request: The request that finished generating tokens.
            cache_block_ids: The block IDs used by the request.

        Returns:
            Whether the request is performing asynchronous saving operations.
        """
        request_id = str(request.request_id)

        # Remove from inflight tracking
        if request_id in self.inflight_requests:
            del self.inflight_requests[request_id]

        return self.leader.request_finished(request_id)

    def update_connector_output(
        self, finished_sending: List[int], finished_recving: List[int]
    ) -> None:
        """
        Update the scheduler with the finished requests.
        """
        print(
            f"Updating connector output with finished_sending: {finished_sending} and finished_recving: {finished_recving}"
        )
        self.leader.update_connector_output(
            set(finished_sending), set(finished_recving)
        )

    def set_handshake_metadata(self, metadata: dict[int, Any]) -> None:
        """
        Register all worker Nova peers and trigger initialization.

        This is called after aggregating handshake metadata from all workers.

        Args:
            metadata: Dictionary mapping rank (int) to NovaPeerMetadata
        """
        register_workers_from_handshake(self.leader, metadata)

    def _create_slot(self, request: "LlmRequest") -> None:
        """Create a slot for the request."""
        request_id = str(request.request_id)

        if self.leader.has_slot(request_id):
            self._update_slot(request_id)
            return

        if bool(request.multimodal_positions):
            raise ValueError("Unsupported request - requires mm extra keys")

        all_token_ids = list(request.get_tokens(0))

        # Create request with tokens embedded (v2 style)
        kv_request = KvbmRequest(
            request_id=request_id,
            tokens=all_token_ids,
            lora_name=None,  # TRTLLM doesn't expose LoRA info here
            salt_hash=None,
            max_tokens=getattr(request, "max_tokens", len(all_token_ids) + 1024),
        )

        self.leader.create_slot(kv_request)

        # Track for decode offload
        self.inflight_requests[request_id] = request

    def _update_slot(self, request_id: str) -> None:
        """
        Synchronize new tokens from the request to the slot.

        This is used for decode offload to track generated tokens.
        """
        request = self.inflight_requests.get(request_id)
        if request is None:
            return

        if not self.leader.has_slot(request_id):
            return

        slot_token_count = self.leader.get_slot_total_tokens(request_id)
        all_tokens = list(request.get_tokens(0))
        request_token_count = len(all_tokens)

        if slot_token_count < request_token_count:
            new_tokens = all_tokens[slot_token_count:]
            self.leader.extend_slot_tokens(request_id, new_tokens)
