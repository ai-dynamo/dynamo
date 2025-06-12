# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from typing import Optional

from vllm.distributed.kv_events import KVCacheEvent
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, PrefixCacheStats
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.request import Request

from dynamo.llm import BlockManager
from dynamo.llm.vllm_integration.kv_cache_utils import KvbmCacheBlocks
from dynamo.llm.vllm_integration.rust import KvbmCacheManager as RustKvbmCacheManager
from dynamo.llm.vllm_integration.rust import KvbmRequest, SlotUpdate


class KvbmCacheManager:
    """
    Implements the vLLM KV cache manager protocol.

    This class is a wrapper around the Rust KvbmCacheManager class.
    It is used to convert the Rust KvbmCacheManager into a Python class
    that can be used in the vLLM KV cache manager protocol.
    """

    def __init__(
        self,
        block_manager: BlockManager,
        log_stats: bool = False,
    ) -> None:
        """
        Initializes the KvbmCacheManager.

        Args:
            block_manager: Python bound Dynamo KV Block Manager (KVBM).
        """
        # pass the python bound KVBM to the Rust KVBM cache manager
        # the rust cache manager will take ownership of the kvbm
        self.cache_manager = RustKvbmCacheManager(block_manager)
        self.block_size = block_manager.block_size()
        self.log_stats = log_stats
        # FIXME: make prefix cache stats conditional on log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.cache_manager.usage()

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KvbmCacheBlocks, int]:
        """
        Get the computed blocks for the request.
        """
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1

        sequence_hashes = self._create_slot(request)

        owned_blocks = self.cache_manager.get_computed_blocks(sequence_hashes)
        block_count = owned_blocks.block_count()

        num_computed_tokens = block_count * self.block_size

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.queries += request.num_tokens
            self.prefix_cache_stats.hits += num_computed_tokens

        print(f"owned_blocks_count: {block_count}")

        return KvbmCacheBlocks(owned_blocks), num_computed_tokens

    def _create_slot(self, request: Request) -> list[int]:
        """Create a slot for the request."""
        if bool(request.mm_positions):
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

        return self.cache_manager.create_slot(request, all_token_ids)

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
        num_draft_tokens: int = 0,
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> Optional[KVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if not self.cache_manager.has_slot(request.request_id):
            self._create_slot(request)

        num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens

        # we need to extract from the request the new tokens to append to the block state
        prev_computed_tokens = self.cache_manager.num_computed_tokens(
            request.request_id
        )
        tokens_to_append = request.all_token_ids[
            prev_computed_tokens:num_computed_tokens
        ]

        print(
            f"request_id: {request.request_id}, num_new_tokens: {num_new_tokens}, num_new_computed_tokens: {num_new_computed_tokens}, tokens_to_append: {len(tokens_to_append)}"
        )

        # take ownership "owned_blocks" of the new computed blocks
        owned_blocks = getattr(new_computed_blocks, "_owned_blocks", None)
        if owned_blocks:
            new_computed_blocks._owned_blocks = None

        slot_update = SlotUpdate(
            request_id=request.request_id,
            request_num_tokens=request.num_tokens,
            request_num_computed_tokens=request.num_computed_tokens,
            tokens_to_append=tokens_to_append,
            num_new_tokens=num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=owned_blocks,
            # TODO(ryan): add support for lookahead blocks
            # comment out for now, otherwise would error out
            # num_lookahead_blocks=num_lookahead_tokens,
            delay_cache_blocks=delay_cache_blocks,
        )

        new_blocks = self.cache_manager.alloctate_slots(slot_update)

        if new_blocks is None:
            return None

        new_blocks = [
            KVCacheBlock(block_id=block_id) for block_id in new_blocks.block_ids()
        ]

        return KVCacheBlocks(blocks=new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that he tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self.cache_manager.free(request.request_id)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        # self.cache_manager.reset_prefix_cache()
        return False

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> list[int]:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state for each kv cache group.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return [0]

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.cache_manager.free_block_hashes(request.request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return []

    def get_block_ids(self, request_id: str) -> list[list[int]]:
        """Get the block ids of a request."""
        return [self.cache_manager.get_block_ids(request_id)]
