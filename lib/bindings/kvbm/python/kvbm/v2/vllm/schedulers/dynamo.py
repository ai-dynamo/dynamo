# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dynamo Scheduler implementation that forwards to vLLM's default scheduler.

This module provides a custom scheduler that acts as a springboard to vLLM's
default scheduler implementation, allowing for future customization while
maintaining compatibility with vLLM's scheduling interface.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple, Union

from vllm.config import VllmConfig

# added to the api in vllm v0.11
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorBase_V1
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.kv_cache_manager import KVCacheConfig
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import SchedulerStats
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

try:
    from kvbm._core.v2 import RustSchedulerState
except ImportError:
    RustSchedulerState = None
    print("Warning: kvbm not available; forwarding all requests to vLLM scheduler")


class DynamoScheduler(SchedulerInterface):
    """
    Custom scheduler that forwards all operations to vLLM's default Scheduler.

    This scheduler acts as a transparent proxy, allowing for future customization
    of scheduling behavior while maintaining full compatibility with vLLM.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        """
        Initialize the DynamoScheduler with a wrapped vLLM Scheduler.

        Args:
            vllm_config: vLLM configuration object
            kv_cache_config: KV cache configuration
            structured_output_manager: Manager for structured outputs
            mm_registry: Multi-modal registry (optional, will use default if None)
            include_finished_set: Whether to include finished requests
            log_stats: Whether to log statistics
        """
        # Create the underlying vLLM scheduler
        self._scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            structured_output_manager=structured_output_manager,
            mm_registry=mm_registry,
            include_finished_set=include_finished_set,
            log_stats=log_stats,
        )

        # Initialize Rust scheduler state if available
        if RustSchedulerState is not None:
            self._rust_scheduler = RustSchedulerState()
            print("DynamoScheduler: Rust scheduler state initialized")
        else:
            self._rust_scheduler = None

    def schedule(self) -> "SchedulerOutput":
        """
        Schedule requests for the next model forward pass.

        Returns:
            SchedulerOutput containing scheduling decisions
        """
        return self._scheduler.schedule()

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

        # Remove finished requests from Rust scheduler
        if self._rust_scheduler is not None and hasattr(
            scheduler_output, "finished_req_ids"
        ):
            try:
                finished_ids = list(scheduler_output.finished_req_ids)
                if finished_ids:
                    self._rust_scheduler.remove_finished_requests(finished_ids)
                    print(
                        f"DynamoScheduler: Removed {len(finished_ids)} finished requests from Rust scheduler"
                    )
            except Exception as e:
                print(
                    f"DynamoScheduler: Error removing finished requests from Rust scheduler: {e}"
                )

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
        # Pass request to Rust scheduler if available
        if self._rust_scheduler is not None:
            try:
                # Extract data available at add_request time
                request_id = request.request_id
                prompt_token_ids = request.prompt_token_ids

                # Pass cache_salt as string - Rust will handle the hashing
                cache_salt = getattr(request, "cache_salt", None)

                # Extract LoRA ID if present
                lora_int_id = None
                if hasattr(request, "lora_request") and request.lora_request:
                    lora_int_id = request.lora_request.lora_int_id

                # Get priority and arrival time
                priority = getattr(request, "priority", 0)
                arrival_time = getattr(request, "arrival_time", 0.0)

                # Add to Rust scheduler (cache_salt is now passed as string)
                self._rust_scheduler.add_request(
                    request_id=request_id,
                    prompt_token_ids=list(prompt_token_ids),  # Convert to list
                    cache_salt=cache_salt,  # Pass as string, Rust converts to u64
                    lora_int_id=lora_int_id,
                    priority=priority,
                    arrival_time=arrival_time,
                )
            except Exception as e:
                print(f"DynamoScheduler: Error adding request to Rust scheduler: {e}")

        # Always add to vLLM scheduler
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
        # Mark as finished in Rust scheduler (doesn't remove them yet)
        if self._rust_scheduler is not None:
            try:
                # Ensure request_ids is a list
                if isinstance(request_ids, str):
                    ids_list = [request_ids]
                else:
                    ids_list = list(request_ids)

                self._rust_scheduler.mark_as_finished(ids_list)
                print(
                    f"DynamoScheduler: Marked {len(ids_list)} requests as finished in Rust scheduler"
                )
            except Exception as e:
                print(
                    f"DynamoScheduler: Error marking requests as finished in Rust scheduler: {e}"
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
        return None
