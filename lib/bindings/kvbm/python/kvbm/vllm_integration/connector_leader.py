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
from kvbm.management import (
    register_clear_pool,
    register_get_cpu_lookup_status,
    register_set_cpu_lookup_disabled,
    start_management_server,
)
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
        world_size = vllm_config.parallel_config.world_size

        leader = KvbmLeader(world_size, drt=self.drt)

        print(f"KvConnectorLeader initialized with engine_id: {engine_id}")
        # Get kv event consolidator endpoints from vllm_config (pre-computed in main.py)
        consolidator_vllm_endpoint = None
        consolidator_output_endpoint = None
        self._consolidator_output_port = None

        if (
            hasattr(vllm_config, "consolidator_endpoints")
            and vllm_config.consolidator_endpoints
        ):
            # Unpack all three endpoints
            # [0]: vllm_endpoint (for consolidator to subscribe to vLLM)
            # [1]: output_bind_endpoint (for consolidator to bind/publish)
            # [2]: output_connect_endpoint (for clients to connect)
            (
                consolidator_vllm_endpoint,
                consolidator_output_endpoint,
                _consolidator_output_connect_endpoint,  # Not needed here
            ) = vllm_config.consolidator_endpoints
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

        # Register the clear_pool callable and start the management HTTP
        # server when KVBM_DEV_MODE is enabled.
        register_clear_pool(self.clear_pool)
        register_set_cpu_lookup_disabled(self.set_cpu_cache_lookup_disabled)
        register_get_cpu_lookup_status(self.get_cpu_cache_lookup_status)
        mgmt_port = start_management_server()
        if mgmt_port is not None:
            print(
                f"[KVBM] Management API available at http://0.0.0.0:{mgmt_port}"
            )

    # KV Connector

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[Optional[int], bool]:
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
                  external KV cache beyond what is already computed,
                  or `None` when lookup is still in progress and should
                  be retried on a later scheduler step.
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

        # In https://github.com/vllm-project/vllm/pull/26388/changes#diff-9eeca590fd99f15621897e559dba39b3ec4e7c2c65ec3c3229711689e008b5f4L732-L736,
        # new_token_ids was changed to return an empty list unless pipeline
        # parallelism is turned on. If needed, which for KVBM it is not needed,
        # KVBM can consult the cached_reqs.all_token_ids to get the token ids
        # for each of the requests. KVBM doesn't consult this field since
        # it holds the token sequence in the _connector.
        for i, req_id in enumerate(cached_reqs.req_ids):
            # new_token_ids may be empty when pipeline parallelism is disabled

            new_token_ids = (
                cached_reqs.new_token_ids[i]
                if i < len(cached_reqs.new_token_ids)
                else []
            )
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
        # note our worker can communication with us oob and we can use that to know
        # ahead of time if the request is finished.
        status = self._connector.request_finished(request.request_id, block_ids)
        return status, None

    # Management API

    def clear_pool(self, pool: str) -> None:
        """Clear (wipe) all KV cache entries from a specific pool.

        Requires KVBM_DEV_MODE=TRUE environment variable.

        This is a destructive operation that:
          1. Drops all in-flight slots (freeing their block references).
          2. Resets the target pool, returning every block to the empty state.

        Args:
            pool: One of "gpu"/"device", "cpu"/"host", or "disk".

        Raises:
            RuntimeError: If KVBM_DEV_MODE is not enabled or pool is invalid.
        """
        self._connector.clear_pool(pool)

    def set_cpu_cache_lookup_disabled(self, disabled: bool) -> None:
        """Enable/disable CPU cache lookup for cache hits (dev-only)."""
        self._connector.set_cpu_cache_lookup_disabled(bool(disabled))

    def get_cpu_cache_lookup_status(self) -> dict:
        """Return CPU cache lookup status for management API."""
        return {
            "disabled": self._connector.get_cpu_cache_lookup_disabled(),
            "dirty": self._connector.cpu_cache_lookup_dirty(),
        }

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

        # extract the critial aspects of the request that effect how the tokens are hashed
        request = KvbmRequest(
            request_id=request.request_id,
            lora_name=request.lora_request.lora_name()
            if request.lora_request
            else None,
            salt_hash=request.cache_salt,
        )

        self._connector.create_slot(request, all_token_ids)
