# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Scheduler implementation with inverted shadow observer pattern.

This module provides a custom scheduler that uses the Rust scheduler as primary,
with vLLM's scheduler running in shadow mode for comparison. Differences between
the two schedulers are printed as loud warnings to stderr.
"""

from __future__ import annotations

import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

# added to the api in vllm v0.11
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheConfig
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from .connector import DynamoConnector
from .output import RustCachedRequestData, RustNewRequestData, RustSchedulerOutput

try:
    from kvbm._core import v2 as kvbm_v2

    ConnectorLeader = kvbm_v2.ConnectorLeader
    RustScheduler = kvbm_v2.RustScheduler
    RustSchedulerConfig = kvbm_v2.SchedulerConfig
    RustRequestStatus = kvbm_v2.RequestStatus
    _RUST_SCHEDULER_AVAILABLE = True
except ImportError:
    ConnectorLeader = None
    RustScheduler = None
    RustSchedulerConfig = None
    RustRequestStatus = None
    _RUST_SCHEDULER_AVAILABLE = False
    print(
        "Warning: kvbm Rust scheduler not available; forwarding all requests to vLLM scheduler"
    )


class DynamoScheduler(SchedulerInterface):
    """
    Scheduler with inverted shadow observer pattern.

    The Rust scheduler is the primary decision maker. vLLM's scheduler runs in
    shadow mode for comparison. Differences are printed as loud warnings.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        """
        Initialize the DynamoScheduler with Rust scheduler as primary.

        Args:
            vllm_config: vLLM configuration object
            kv_cache_config: KV cache configuration
            structured_output_manager: Manager for structured outputs
            block_size: Block size for KV cache
            mm_registry: Multi-modal registry (optional, will use default if None)
            include_finished_set: Whether to include finished requests
            log_stats: Whether to log statistics
        """
        # Create the underlying vLLM scheduler (shadow mode)
        self._scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            block_size=block_size,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        # Request tracking for reconstructing output data
        # Maps req_id -> Request object (for mm_features, sampling_params, etc.)
        self._requests: Dict[str, Request] = {}
        # Track output tokens per request (for all_token_ids in cached requests)
        self._output_tokens: Dict[str, List[int]] = {}
        # Track which requests were scheduled in the previous step
        self._prev_scheduled_req_ids: Set[str] = set()

        # Initialize Rust scheduler if available
        if _RUST_SCHEDULER_AVAILABLE:
            try:
                # Get total blocks from KV cache config if available
                total_blocks = None
                if hasattr(kv_cache_config, "num_blocks"):
                    total_blocks = kv_cache_config.num_blocks
                elif hasattr(kv_cache_config, "total_num_blocks"):
                    total_blocks = kv_cache_config.total_num_blocks

                # Get max_seq_len from model config
                max_seq_len = getattr(vllm_config.model_config, "max_model_len", 8192)

                # Get max_prefill_chunk_size from scheduler config (may be None)
                max_prefill_chunk_size = getattr(
                    vllm_config.scheduler_config, "max_prefill_tokens", None
                )

                # Create KVConnector for the Scheduler. Note that each Worker
                # will have a corresponding KVConnector with Role=WORKER.
                # KV Connector pushes/pull of remote KVs for P/D and offloading.
                self.connector = None
                self.connector_prefix_cache_stats: PrefixCacheStats | None = None
                if self.vllm_config.kv_transfer_config is not None:
                    assert (
                        not self.is_encoder_decoder
                    ), "Encoder-decoder models are not currently supported with KV connectors"
                    self.connector = KVConnectorFactory.create_connector(
                        config=self.vllm_config,
                        role=KVConnectorRole.SCHEDULER,
                        kv_cache_config=self.kv_cache_config,
                    )
                    if self.log_stats:
                        self.connector_prefix_cache_stats = PrefixCacheStats()

                # Extract ConnectorLeader if using DynamoConnector
                # This allows the Rust scheduler to use the connector for
                # intelligent eviction and KV cache offloading
                connector_leader = None
                if (
                    isinstance(self.connector, DynamoConnector)
                    and self.connector._scheduler is not None
                ):
                    connector_leader = self.connector._scheduler.leader

                # Create Rust scheduler config from vLLM config
                # Required fields (from vLLM framework) must be provided explicitly
                # Optional fields use None to get Rust defaults
                rust_config = RustSchedulerConfig(
                    max_seq_len=max_seq_len,
                    max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
                    max_num_seqs=vllm_config.scheduler_config.max_num_seqs,
                    block_size=block_size,
                    enable_prefix_caching=vllm_config.cache_config.enable_prefix_caching,
                    enable_chunked_prefill=vllm_config.scheduler_config.enable_chunked_prefill,
                    max_prefill_chunk_size=max_prefill_chunk_size,
                    # Optional fields - use None to get Rust defaults
                    enable_projection=None,  # Default: False
                    projection_lookahead=None,  # Default: 2 * block_size
                    min_guaranteed_blocks=None,  # Default: 3
                    total_blocks=total_blocks,
                )
                self._rust_scheduler = RustScheduler(
                    rust_config, connector=connector_leader
                )
                print(
                    f"DynamoScheduler: Rust scheduler initialized (total_blocks={total_blocks}, max_seq_len={max_seq_len}, has_connector={connector_leader is not None})"
                )
            except Exception as e:
                print(f"DynamoScheduler: Failed to initialize Rust scheduler: {e}")
                self._rust_scheduler = None
        else:
            self._rust_scheduler = None

    def schedule(self) -> "SchedulerOutput":
        """
        Schedule requests for the next model forward pass.

        Uses Rust scheduler as primary, vLLM scheduler as shadow for comparison.
        Prints loud warnings when the two schedulers disagree.

        Returns:
            SchedulerOutput containing scheduling decisions
        """
        # If Rust scheduler is not available, fall back to vLLM
        if self._rust_scheduler is None:
            return self._scheduler.schedule()

        try:
            # Get vLLM's schedule first to learn about finished requests
            # (vLLM tracks completion internally - EOS token, max tokens, etc.)
            vllm_output = self._scheduler.schedule()

            # Sync finished requests to Rust BEFORE it schedules
            # This ensures Rust doesn't try to schedule already-finished requests
            if vllm_output.finished_req_ids:
                for req_id in vllm_output.finished_req_ids:
                    # Clean up our tracking
                    self._requests.pop(req_id, None)
                    self._output_tokens.pop(req_id, None)
                    self._prev_scheduled_req_ids.discard(req_id)
                # Tell Rust these requests are done
                self._rust_scheduler.finish_requests(
                    list(vllm_output.finished_req_ids),
                    RustRequestStatus.finished_stopped(),
                )

            # Now get Rust scheduler decision (primary)
            rust_output_dict = self._rust_scheduler.schedule()
            rust_output = self._rust_output_to_scheduler_output(rust_output_dict)

            # Use vLLM's finished_req_ids (vLLM tracks completion status, not Rust)
            rust_output.finished_req_ids = vllm_output.finished_req_ids

            # Compare scheduling decisions (not finished_req_ids - that's completion tracking)
            self._compare_outputs(rust_output, vllm_output)

            # Update tracking for next iteration
            self._prev_scheduled_req_ids = set(rust_output.num_scheduled_tokens.keys())

            # Return Rust scheduler's decision with vLLM's completion info
            return rust_output

        except Exception as e:
            print(f"DynamoScheduler: Rust schedule() failed: {e}", file=sys.stderr)
            print("DynamoScheduler: Falling back to vLLM scheduler", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            return self._scheduler.schedule()

    def _rust_output_to_scheduler_output(
        self, rust_output: dict
    ) -> RustSchedulerOutput:
        """Convert Rust scheduler dict to RustSchedulerOutput."""
        # Build new requests list
        new_reqs = []
        for req_data in rust_output.get("scheduled_new_reqs", []):
            req_id = req_data["req_id"]
            original = self._requests.get(req_id)

            # Convert block_ids: list[list[int]] -> tuple[list[int], ...]
            block_ids_raw = req_data.get("block_ids", [[]])
            block_ids = tuple(list(b) for b in block_ids_raw)

            new_reqs.append(
                RustNewRequestData(
                    req_id=req_id,
                    prompt_token_ids=list(req_data.get("prompt_token_ids", [])),
                    block_ids=block_ids,
                    num_computed_tokens=req_data.get("num_computed_tokens", 0),
                    mm_features=original.mm_features if original else [],
                    sampling_params=original.sampling_params if original else None,
                    pooling_params=original.pooling_params if original else None,
                    lora_request=original.lora_request if original else None,
                    prompt_embeds=original.prompt_embeds if original else None,
                )
            )

        # Build cached requests
        cached_raw = rust_output.get("scheduled_cached_reqs", {})
        cached_req_ids = cached_raw.get("req_ids", [])

        # Build resumed_req_ids from resumed_from_preemption flags
        resumed_flags = cached_raw.get("resumed_from_preemption", [])
        resumed_req_ids: Set[str] = set()
        for i, req_id in enumerate(cached_req_ids):
            if i < len(resumed_flags) and resumed_flags[i]:
                resumed_req_ids.add(req_id)

        # Build new_block_ids: list[list[list[int]] | None] -> list[tuple[list[int], ...] | None]
        new_block_ids_raw = cached_raw.get("new_block_ids", [])
        new_block_ids: List[Tuple[List[int], ...] | None] = []
        for bid in new_block_ids_raw:
            if bid is None:
                new_block_ids.append(None)
            else:
                new_block_ids.append(tuple(list(b) for b in bid))

        # Build all_token_ids for requests not scheduled in previous step
        all_token_ids: Dict[str, List[int]] = {}
        for req_id in cached_req_ids:
            if req_id not in self._prev_scheduled_req_ids:
                # Include prompt + output tokens
                original = self._requests.get(req_id)
                if original:
                    all_tokens = list(original.prompt_token_ids)
                    all_tokens.extend(self._output_tokens.get(req_id, []))
                    all_token_ids[req_id] = all_tokens

        # Build num_output_tokens
        num_output_tokens = [
            len(self._output_tokens.get(req_id, [])) for req_id in cached_req_ids
        ]

        cached_reqs = RustCachedRequestData(
            req_ids=cached_req_ids,
            resumed_req_ids=resumed_req_ids,
            resumed_from_preemption=resumed_flags,
            new_token_ids=cached_raw.get("new_token_ids", [[] for _ in cached_req_ids]),
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=cached_raw.get("num_computed_tokens", []),
            num_output_tokens=num_output_tokens,
        )

        # num_common_prefix_blocks needs at least one element (one per KV cache group)
        # Default to [0] if not provided, meaning no common prefix blocks
        num_common_prefix_blocks = rust_output.get("num_common_prefix_blocks", None)
        if num_common_prefix_blocks is None or len(num_common_prefix_blocks) == 0:
            num_common_prefix_blocks = [
                0
            ]  # Default: 1 KV cache group with 0 common prefix

        return RustSchedulerOutput(
            scheduled_new_reqs=new_reqs,
            scheduled_cached_reqs=cached_reqs,
            num_scheduled_tokens=rust_output.get("num_scheduled_tokens", {}),
            total_num_scheduled_tokens=rust_output.get("total_num_scheduled_tokens", 0),
            finished_req_ids=set(rust_output.get("finished_req_ids", [])),
            scheduled_spec_decode_tokens=rust_output.get(
                "scheduled_spec_decode_tokens", {}
            ),
            scheduled_encoder_inputs=rust_output.get("scheduled_encoder_inputs", {}),
            num_common_prefix_blocks=num_common_prefix_blocks,
            free_encoder_mm_hashes=rust_output.get("free_encoder_mm_hashes", []),
        )

    @staticmethod
    def _count_blocks(block_ids: Tuple[List[int], ...] | None) -> int:
        """Count total blocks across all KV cache groups."""
        if block_ids is None:
            return 0
        return sum(len(group) for group in block_ids)

    def _compare_outputs(
        self, rust: RustSchedulerOutput, vllm: SchedulerOutput
    ) -> None:
        """
        Compare scheduler outputs and print loud warnings on differences.

        Note: Block IDs are allowed to differ (Rust has its own allocator),
        but block COUNTS should match.
        """
        differences = []

        # Compare total scheduled tokens
        if rust.total_num_scheduled_tokens != vllm.total_num_scheduled_tokens:
            differences.append(
                f"total_num_scheduled_tokens: Rust={rust.total_num_scheduled_tokens} "
                f"vs vLLM={vllm.total_num_scheduled_tokens}"
            )

        # Compare scheduled request IDs
        rust_req_ids = set(rust.num_scheduled_tokens.keys())
        vllm_req_ids = set(vllm.num_scheduled_tokens.keys())
        if rust_req_ids != vllm_req_ids:
            only_rust = rust_req_ids - vllm_req_ids
            only_vllm = vllm_req_ids - rust_req_ids
            if only_rust:
                differences.append(f"requests only in Rust: {only_rust}")
            if only_vllm:
                differences.append(f"requests only in vLLM: {only_vllm}")

        # Compare per-request token counts
        for req_id in rust_req_ids & vllm_req_ids:
            rust_tokens = rust.num_scheduled_tokens[req_id]
            vllm_tokens = vllm.num_scheduled_tokens[req_id]
            if rust_tokens != vllm_tokens:
                differences.append(
                    f"tokens for {req_id}: Rust={rust_tokens} vs vLLM={vllm_tokens}"
                )

        # Compare new vs cached request splits
        rust_new_ids = {r.req_id for r in rust.scheduled_new_reqs}
        vllm_new_ids = {r.req_id for r in vllm.scheduled_new_reqs}
        if rust_new_ids != vllm_new_ids:
            differences.append(
                f"new_req_ids: Rust={rust_new_ids} vs vLLM={vllm_new_ids}"
            )

        rust_cached_ids = set(rust.scheduled_cached_reqs.req_ids)
        vllm_cached_ids = set(vllm.scheduled_cached_reqs.req_ids)
        if rust_cached_ids != vllm_cached_ids:
            differences.append(
                f"cached_req_ids: Rust={rust_cached_ids} vs vLLM={vllm_cached_ids}"
            )

        # Compare block COUNTS for new requests (not exact IDs - those can differ)
        rust_new_by_id = {r.req_id: r for r in rust.scheduled_new_reqs}
        vllm_new_by_id = {r.req_id: r for r in vllm.scheduled_new_reqs}
        for req_id in rust_new_ids & vllm_new_ids:
            rust_block_count = self._count_blocks(rust_new_by_id[req_id].block_ids)
            vllm_block_count = self._count_blocks(vllm_new_by_id[req_id].block_ids)
            if rust_block_count != vllm_block_count:
                differences.append(
                    f"block_count for new req {req_id}: "
                    f"Rust={rust_block_count} vs vLLM={vllm_block_count}"
                )

        # Compare block COUNTS for cached requests' new_block_ids
        rust_cached = rust.scheduled_cached_reqs
        vllm_cached = vllm.scheduled_cached_reqs
        for i, req_id in enumerate(rust_cached.req_ids):
            if req_id in vllm_cached_ids:
                vllm_idx = vllm_cached.req_ids.index(req_id)
                rust_new_blocks = (
                    rust_cached.new_block_ids[i]
                    if i < len(rust_cached.new_block_ids)
                    else None
                )
                vllm_new_blocks = (
                    vllm_cached.new_block_ids[vllm_idx]
                    if vllm_idx < len(vllm_cached.new_block_ids)
                    else None
                )
                rust_count = self._count_blocks(rust_new_blocks)
                vllm_count = self._count_blocks(vllm_new_blocks)
                if rust_count != vllm_count:
                    differences.append(
                        f"new_block_count for cached req {req_id}: "
                        f"Rust={rust_count} vs vLLM={vllm_count}"
                    )

        # Note: finished_req_ids is NOT compared - it's completion tracking handled by vLLM,
        # not a scheduling decision. We sync it from vLLM to Rust before scheduling.

        # Print loud warnings if there are differences
        if differences:
            print("=" * 70, file=sys.stderr)
            print("!!! SCHEDULER DIVERGENCE DETECTED !!!", file=sys.stderr)
            print("=" * 70, file=sys.stderr)
            for diff in differences:
                print(f"  {diff}", file=sys.stderr)
            print("=" * 70, file=sys.stderr)

    def update_from_output(
        self,
        scheduler_output: "SchedulerOutput",
        model_runner_output: "ModelRunnerOutput",
    ) -> Dict[int, "EngineCoreOutputs"]:
        """
        Update scheduler state after model processing.

        Args:
            scheduler_output: Output from the schedule() method
            model_runner_output: Output from the model runner

        Returns:
            Dictionary mapping request IDs to engine core outputs
        """
        result = self._scheduler.update_from_output(
            scheduler_output, model_runner_output
        )

        # Extract output tokens per request
        output_tokens: Dict[str, List[int]] = {}
        if hasattr(model_runner_output, "sampled_token_ids"):
            for i, req_id in enumerate(model_runner_output.req_ids):
                if i < len(model_runner_output.sampled_token_ids):
                    tokens = model_runner_output.sampled_token_ids[i]
                    if hasattr(tokens, "tolist"):
                        tokens = tokens.tolist()
                    output_tokens[req_id] = list(tokens)

        # Track output tokens locally for all_token_ids reconstruction
        for req_id, tokens in output_tokens.items():
            if req_id in self._output_tokens:
                self._output_tokens[req_id].extend(tokens)

        # Update Rust scheduler with output tokens
        if self._rust_scheduler is not None:
            try:
                # Extract finished request IDs
                finished_ids = (
                    list(scheduler_output.finished_req_ids)
                    if hasattr(scheduler_output, "finished_req_ids")
                    else []
                )
                self._rust_scheduler.update_from_output(finished_ids, output_tokens)
            except Exception as e:
                print(f"DynamoScheduler: Error updating Rust scheduler: {e}")

        return result

    def update_draft_token_ids(
        self,
        draft_token_ids: "DraftTokenIds",
    ) -> None:
        """
        Update draft token IDs for scheduled requests.

        Args:
            draft_token_ids: Draft token IDs to update
        """
        self._scheduler.update_draft_token_ids(draft_token_ids)

    def add_request(self, request: "Request") -> None:
        """
        Add a new request to the scheduler.

        Args:
            request: Request object to add to the scheduler
        """
        # Store request for output reconstruction
        self._requests[request.request_id] = request
        self._output_tokens[request.request_id] = []

        # Pass request to Rust scheduler if available
        if self._rust_scheduler is not None:
            try:
                request_id = request.request_id
                prompt_token_ids = list(request.prompt_token_ids)
                self._rust_scheduler.add_request(request_id, prompt_token_ids)
            except Exception as e:
                print(f"DynamoScheduler: Error adding request to Rust scheduler: {e}")

        # Always add to vLLM scheduler (shadow mode)
        self._scheduler.add_request(request)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status: "RequestStatus",
    ) -> None:
        """
        Mark requests as finished.

        Args:
            request_ids: Request ID(s) to mark as finished
            finished_status: The finish status for the requests
        """
        # Ensure request_ids is a list
        if isinstance(request_ids, str):
            ids_list = [request_ids]
        else:
            ids_list = list(request_ids)

        # Clean up stored request data
        for req_id in ids_list:
            self._requests.pop(req_id, None)
            self._output_tokens.pop(req_id, None)
            self._prev_scheduled_req_ids.discard(req_id)

        # Mark as finished in Rust scheduler
        if self._rust_scheduler is not None:
            try:
                # Map vLLM status to Rust status
                rust_status = RustRequestStatus.finished_stopped()
                self._rust_scheduler.finish_requests(ids_list, rust_status)
            except Exception as e:
                print(
                    f"DynamoScheduler: Error finishing requests in Rust scheduler: {e}"
                )

        # Always call vLLM scheduler to handle the actual state transitions
        self._scheduler.finish_requests(request_ids, finished_status)

    def get_num_unfinished_requests(self) -> int:
        """
        Get the number of unfinished requests.

        Returns:
            Number of unfinished requests in the scheduler
        """
        return self._scheduler.get_num_unfinished_requests()

    def has_finished_requests(self) -> bool:
        """
        Check if there are any finished requests.

        Returns:
            True if there are finished requests, False otherwise
        """
        return self._scheduler.has_finished_requests()

    def reset_prefix_cache(self) -> bool:
        """
        Reset the prefix cache.

        Returns:
            True if cache was reset successfully
        """
        return self._scheduler.reset_prefix_cache()

    def get_request_counts(self) -> Tuple[int, int]:
        """
        Get counts of requests in different states.

        Returns:
            Tuple of (waiting_count, running_count)
        """
        return self._scheduler.get_request_counts()

    def make_stats(self) -> Optional[SchedulerStats]:
        """
        Generate statistics about the scheduler's current state.

        Returns:
            SchedulerStats object or None
        """
        return self._scheduler.make_stats()

    def shutdown(self) -> None:
        """
        Shutdown the scheduler and clean up resources.
        """
        self._scheduler.shutdown()

    # new in vllm v0.11
    def get_kv_connector(self) -> Optional[KVConnectorBase_V1]:
        return self.connector

    # new in vllm v0.12
    def get_grammar_bitmask(self, scheduler_output: "SchedulerOutput"):
        """
        Get grammar bitmask for structured output generation.

        Args:
            scheduler_output: Output from the schedule() method

        Returns:
            Grammar bitmask or None if not applicable
        """
        return self._scheduler.get_grammar_bitmask(scheduler_output)
