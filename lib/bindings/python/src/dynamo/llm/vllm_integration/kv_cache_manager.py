# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM KV cache manager protocol.
"""

from typing import Optional

from vllm.v1.core.kv_cache_manager import KVCacheBlocks
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

    def __init__(self, block_manager: BlockManager):
        """
        Initializes the KvbmCacheManager.

        Args:
            block_manager: Python bound Dynamo KV Block Manager (KVBM).
        """
        # pass the python bound KVBM to the Rust KVBM cache manager
        # the rust cache manager will take ownership of the kvbm
        self.cache_manager = RustKvbmCacheManager(block_manager)
        self.block_size = block_manager.block_size()

    def get_computed_blocks(self, request: Request) -> tuple[KvbmCacheBlocks, int]:
        """
        Get the computed blocks for the request.
        """
        if bool(request.mm_positions):
            raise ValueError("Unsupported request - requires mm extra keys")

        # extract the critial aspects of the request that effect how the tokens are hashed
        request = KvbmRequest(
            request_id=request.request_id,
            lora_name=request.lora_request.lora_name()
            if request.lora_request
            else None,
            salt_hash=request.cache_salt,
        )

        # todo(vllm): determine if this call should be idempotent or if it should fail
        # if the slot already exists
        sequence_hashes = self.cache_manager.create_slot(request, request.all_token_ids)

        owned_blocks = self.cache_manager.get_computed_blocks(sequence_hashes)
        block_count = owned_blocks.block_count()

        return KvbmCacheBlocks(owned_blocks), block_count

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: Optional[KVCacheBlocks] = None,
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

        # we need to extract from the request the new tokens to append to to the block state
        num_sequence_tokens = self.cache_manager.num_sequence_tokens(request.request_id)
        tokens_to_append = request.all_token_ids[
            num_sequence_tokens : request.num_tokens
        ]

        slot_update = SlotUpdate(
            request_id=request.request_id,
            request_num_tokens=request.num_tokens,
            tokens_to_append=tokens_to_append,
            num_new_tokens=num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_blocks=num_lookahead_tokens,
            delay_cache_blocks=delay_cache_blocks,
        )

        new_blocks = self.cache_manager.alloctate_slots(slot_update)

        if new_blocks is None:
            return None

        new_blocks = [
            KVCacheBlock(block_id=block.block_id) for block in new_blocks.block_ids()
        ]

        return KVCacheBlocks(blocks=new_blocks)
