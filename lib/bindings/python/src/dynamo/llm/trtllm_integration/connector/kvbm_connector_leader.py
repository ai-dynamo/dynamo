# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from typing import List

from tensorrt_llm._torch.pyexecutor.kv_cache_connector import (
    KvCacheConnectorScheduler,
    SchedulerOutput,
)
from tensorrt_llm.bindings.internal.batch_manager import LlmRequest
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

from dynamo.llm import KvbmLeader
from dynamo.llm.trtllm_integration.rust import KvbmRequest
from dynamo.llm.trtllm_integration.rust import (
    KvConnectorLeader as RustKvConnectorLeader,
)
from dynamo.llm.trtllm_integration.rust import SchedulerOutput as RustSchedulerOutput
from dynamo.runtime import DistributedRuntime


class DynamoKVBMConnectorLeader(KvCacheConnectorScheduler):
    def __init__(self, llm_args: TorchLlmArgs):
        super().__init__(llm_args)
        self.drt = DistributedRuntime.detached()

        mappings = self._llm_args.parallel_config.to_mapping()

        world_size = mappings.world_size
        self.block_size = self._llm_args.kv_cache_config.tokens_per_block

        # Set bytes_per_block to 0, because we will retrieve the actual value from the worker side.
        leader = KvbmLeader(world_size, drt=self.drt)

        print(f"KvConnectorLeader initialized with rank: {mappings.rank}")
        self._connector = RustKvConnectorLeader(
            mappings.rank, self.drt, self.block_size, leader
        )

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> bytes:
        """
        Build the metadata for the worker.
        This is called by the KV Cache Manager when adding a sequence.
        Args:
            scheduler_output: The data for all inflight requests.
        Returns:
            The metadata for the workers.
        """
        output = RustSchedulerOutput()

        for req in scheduler_output.new_requests:
            output.add_new_request(
                str(req.request_id),
                req.new_tokens,
                req.new_block_ids,
                req.computed_position,
            )

        resumed_from_preemption = False
        for req in scheduler_output.cached_requests:
            output.add_cached_request(
                str(req.request_id),
                resumed_from_preemption,
                req.new_tokens,
                req.new_block_ids,
                req.computed_position,
            )

        return self._connector.build_connector_metadata(output)

    def get_num_new_matched_tokens(
        self, request: LlmRequest, num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        Get the number of tokens that can be loaded from remote KV cache.
        This does not include the tokens already matched on device (indicated by `num_computed_tokens`).
        Args:
            request: The request to get the number of tokens for.
            num_computed_tokens: The number of tokens already matched on device.
        Returns:
            The number of tokens that can be loaded from remote KV cache.
            Whether the tokens will be loaded asynchronously.
        """
        self._create_slot(request)
        return self._connector.get_num_new_matched_tokens(
            str(request.request_id),
            len(request.get_tokens(0)),
            num_computed_tokens,
        )

    def update_state_after_alloc(self, request: LlmRequest, block_ids: List[int]):
        """
        Called after get_num_new_matched_tokens is called to provide the block ids to the scheduler.
        Args:
            request: The request that was allocated resources.
            block_ids: The KV cacheblock IDs that were allocated.
        """
        self._connector.update_state_after_alloc(
            str(request.request_id), block_ids, request.context_current_position
        )

    def request_finished(self, request: LlmRequest, cache_block_ids: list[int]) -> bool:
        """
        Called when a request is finished generating tokens.
        Args:
            request: The request that finished generating tokens.
        Returns:
            Whether the request is performing asynchronous saving operations.
            If true, this indicates that the kv cache manager should wait to deallocate the blocks until the saving has completed (determined by `get_finished` on the workers).
        """
        is_async_saving = self._connector.request_finished(
            str(request.request_id), cache_block_ids
        )
        return is_async_saving

    def _create_slot(self, request: LlmRequest) -> None:
        """Create a slot for the request"""

        if self._connector.has_slot(str(request.request_id)):
            return None

        if bool(request.multimodal_positions):
            raise ValueError("Unsupported request - requires mm extra keys")

        all_token_ids = request.get_tokens(0)

        # extract the critial aspects of the request that effect how the tokens are hashed
        request = KvbmRequest(
            request_id=str(request.request_id), lora_name=None, salt_hash=None
        )

        self._connector.create_slot(request, all_token_ids)
