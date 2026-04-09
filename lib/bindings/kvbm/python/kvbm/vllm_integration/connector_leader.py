# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import Request

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request


# from kvbm.vllm_integration.kv_cache_utils import KvbmCacheBlocks
# from kvbm.vllm_integration.rust import BlockManager, KvbmRequest
# from kvbm.vllm_integration.rust import KvConnectorLeader as RustKvConnectorLeader
# from kvbm.vllm_integration.rust import (
#     KvConnectorMetadata as RustKvConnectorMetadata,
# )
# from kvbm.vllm_integration.rust import SchedulerOutput as RustSchedulerOutput

from kvbm import KvbmLeader
from kvbm.utils import is_dyn_runtime_enabled
from kvbm.vllm_integration.rust import KvbmRequest
from kvbm.vllm_integration.rust import KvConnectorLeader as RustKvConnectorLeader
from kvbm.vllm_integration.rust import SchedulerOutput as RustSchedulerOutput

DistributedRuntime = None
if is_dyn_runtime_enabled():
    from dynamo.runtime import DistributedRuntime


class DynamoConnectorMetadata(KVConnectorMetadata):
    def __init__(self, metadata: bytes):
        assert isinstance(metadata, bytes)
        self.metadata = metadata


class KvConnectorLeader:
    """
    Implements the vLLM KV cache manager protocol.

    This class is a wrapper around the Rust KvbmCacheManager class.
    It is used to convert the Rust KvbmCacheManager into a Python class
    that can be used in the vLLM KV cache manager protocol.
    """

    def __init__(self, vllm_config: "VllmConfig", engine_id: str, **kwargs):
        drt: Optional[object] = kwargs.get("drt")

        if drt is None and is_dyn_runtime_enabled():
            drt = DistributedRuntime.detached()
        else:
            drt = None

        self.drt = drt
        self.vllm_config = vllm_config
        self._slot_seq_len: dict[str, int] = {}
        world_size = vllm_config.parallel_config.world_size

        leader = KvbmLeader(world_size, drt=self.drt)

        print(f"KvConnectorLeader initialized with engine_id: {engine_id}")
        # Get kv event consolidator endpoints from vllm_config (pre-computed in main.py)
        consolidator_vllm_endpoint = None
        consolidator_output_endpoint = None
        self._consolidator_output_port = None

        _consolidator_eps = vllm_config.additional_config.get("consolidator_endpoints")
        if _consolidator_eps:
            # Unpack all three endpoints
            # [0]: vllm_endpoint (for consolidator to subscribe to vLLM)
            # [1]: output_bind_endpoint (for consolidator to bind/publish)
            # [2]: output_connect_endpoint (for clients to connect)
            (
                consolidator_vllm_endpoint,
                consolidator_output_endpoint,
                _consolidator_output_connect_endpoint,  # Not needed here
            ) = _consolidator_eps
            self._consolidator_output_port = int(
                consolidator_output_endpoint.split(":")[-1]
            )

            # Pass endpoints to Rust
            self._connector = RustKvConnectorLeader(
                engine_id,
                self.drt,
                vllm_config.cache_config.block_size,
                leader,
                consolidator_vllm_endpoint=consolidator_vllm_endpoint,
                consolidator_output_endpoint=consolidator_output_endpoint,
            )
        else:
            # No kv event consolidator - pass None to Rust
            self._connector = RustKvConnectorLeader(
                engine_id,
                self.drt,
                vllm_config.cache_config.block_size,
                leader,
                consolidator_vllm_endpoint=None,
                consolidator_output_endpoint=None,
            )

    # KV Connector

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """
        Get number of new tokens that can be loaded from the
        external KV cache beyond the num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded from the
                  external KV cache beyond what is already computed.
                - `True` if external KV cache tokens will be loaded
                  asynchronously (between scheduler steps).
        """
        self._create_slot(request)
        return self._connector.get_num_new_matched_tokens(
            request.request_id,
            request.num_tokens,
            num_computed_tokens,
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        block_ids = blocks.get_block_ids()[0]
        self._connector.update_state_after_alloc(
            request.request_id, block_ids, num_external_tokens
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> bytes:
        """
        Build the connector metadata for this step.

        This function should NOT modify fields in the scheduler_output.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """
        output = RustSchedulerOutput()

        for req in scheduler_output.scheduled_new_reqs:
            output.add_new_request(
                req.req_id,
                req.prompt_token_ids,
                req.block_ids[0],
                req.num_computed_tokens,
            )

        # In vLLM 0.14.0+, resumed_from_preemption was changed to resumed_req_ids (a set)
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids

        # If A == B, and A == C, then B == C
        cached_reqs = scheduler_output.scheduled_cached_reqs
        assert len(cached_reqs.req_ids) == len(
            cached_reqs.new_block_ids
        ), "Number of cached req_ids doesn't match the number of cached new_block_ids"
        assert len(cached_reqs.req_ids) == len(
            cached_reqs.num_computed_tokens
        ), "Number of cached req_ids doesn't match the number of cached num_computed_tokens"

        # vLLM >= 0.16 returns empty new_token_ids when pipeline parallelism
        # is disabled.  KVBM needs the decode tokens to grow the internal
        # TokenBlockSequence (required for correct hash computation during
        # offloading).  When new_token_ids is empty, derive the missing
        # tokens from cached_reqs.all_token_ids (a dict[str, list[int]]
        # keyed by request_id, only populated for requests that were NOT
        # scheduled in the prior step — e.g. resumed from preemption).
        # For continuously-scheduled requests the tokens are fed later via
        # request_finished where the full sequence is available.
        for i, req_id in enumerate(cached_reqs.req_ids):
            new_token_ids = (
                cached_reqs.new_token_ids[i]
                if i < len(cached_reqs.new_token_ids)
                else []
            )

            if not new_token_ids and hasattr(cached_reqs, "all_token_ids"):
                all_ids = cached_reqs.all_token_ids.get(req_id)
                if all_ids is not None:
                    last_fed = self._slot_seq_len.get(req_id, 0)
                    if len(all_ids) > last_fed:
                        new_token_ids = list(all_ids[last_fed:])
                        self._slot_seq_len[req_id] = len(all_ids)
            new_block_ids = cached_reqs.new_block_ids[i]
            num_computed_tokens = cached_reqs.num_computed_tokens[i]

            resumed_from_preemption = req_id in resumed_req_ids
            if new_block_ids is not None:
                output.add_cached_request(
                    request_id=req_id,
                    resumed_from_preemption=resumed_from_preemption,
                    new_token_ids=new_token_ids,
                    new_block_ids=new_block_ids[0],
                    num_computed_tokens=num_computed_tokens,
                )
            else:
                output.add_cached_request(
                    request_id=req_id,
                    resumed_from_preemption=resumed_from_preemption,
                    new_token_ids=new_token_ids,
                    new_block_ids=[],
                    num_computed_tokens=num_computed_tokens,
                )

        output.add_num_scheduled_tokens(scheduler_output.num_scheduled_tokens)

        assert (
            scheduler_output.total_num_scheduled_tokens
            == output.get_num_scheduled_tokens()
        ), "Total number of scheduled tokens does not match"

        return self._connector.build_connector_metadata(output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """
        Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.
        """
        # Feed any remaining decode tokens so the Rust TokenBlockSequence
        # has the full sequence for correct hash computation during the
        # flush-on-finish offload.  vLLM's scheduler does not propagate
        # decode token IDs on every step (only for PP or re-scheduled reqs),
        # so we backfill them here from request.all_token_ids.
        req_id = request.request_id
        all_ids = request.all_token_ids
        last_fed = self._slot_seq_len.get(req_id, 0)
        if len(all_ids) > last_fed:
            remaining = list(all_ids[last_fed:])
            self._connector.feed_tokens(req_id, remaining)

        status = self._connector.request_finished(req_id, block_ids)
        self._slot_seq_len.pop(req_id, None)
        return status, None

    # Utility functions

    def _create_slot(self, request: Request) -> None:
        """Create a slot for the request"""

        if self._connector.has_slot(request.request_id):
            return None

        if bool(getattr(request, "mm_features", None)) or bool(
            getattr(request, "mm_positions", None)
        ):
            raise ValueError("Unsupported request - requires mm extra keys")

        all_token_ids = request.all_token_ids
        req_id = request.request_id

        # extract the critial aspects of the request that effect how the tokens are hashed
        request = KvbmRequest(
            request_id=req_id,
            lora_name=request.lora_request.lora_name()
            if request.lora_request
            else None,
            salt_hash=request.cache_salt,
        )

        self._slot_seq_len[req_id] = len(all_token_ids)
        self._connector.create_slot(request, all_token_ids)
