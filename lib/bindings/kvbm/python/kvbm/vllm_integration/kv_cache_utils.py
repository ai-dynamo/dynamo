# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Implementation of vLLM protocols for KV cache utility objects.
"""

from __future__ import annotations

from typing import List

from kvbm.vllm_integration.rust import BlockState, BlockStates, KvbmBlockList
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock

# from vllm.logger import init_logger
# logger = init_logger(__name__)


class KvbmCacheBlocks:
    """
    Implements the KVCacheBlocksProtocol interface.
    """

    def __init__(self, blocks: KvbmBlockList):
        self._blocks = [
            KVCacheBlock(
                block_id=blocks.get_block_id(i), _block_hash=blocks.get_block_hash(i)
            )
            for i in range(blocks.block_count())
        ]
        self._owned_blocks = blocks

    @property
    def blocks(self) -> List[KVCacheBlock]:
        """
        Returns the list of KVCacheBlock objects.
        """
        return self._blocks

    def get_block_ids(self) -> list[list[int]]:
        """
        Returns the list of block IDs.
        """
        return [[block.block_id for block in self.blocks]]

    def get_unhashed_block_ids(self) -> list[int]:
        """
        Returns the list of unhashed block IDs.
        """
        return [block.block_id for block in self.blocks if block.block_hash is None]

    def __add__(self, other: "KvbmCacheBlocks") -> "KvbmCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        # This is a disgusting hack to get this to work nicely with vLLM.
        return None

    @classmethod
    def create_empty(cls) -> "KvbmCacheBlocks":
        """Creates a new KVCacheBlocks instance with no blocks."""
        raise NotImplementedError("create_empty not implemented")

    def __len__(self):
        return len(self._blocks)


def convert_kv_cache_block(block: KVCacheBlock) -> BlockState:
    """
    Converts a KVCacheBlock object into a BlockState object.
    """
    block_hash = block.block_hash()
    if block_hash is None:
        return BlockState(block_id=block.block_id, tokens=None)
    else:
        return BlockState(
            block_id=block.block_id, tokens=[t for t in block_hash.tokens_ids]
        )


def convert_kv_cache_blocks(blocks: KVCacheBlocks) -> BlockStates:
    """
    Converts a KVCacheBlocks object into a BlockStates object.
    """
    states = BlockStates()
    for block in blocks.blocks:
        states.push_back(convert_kv_cache_block(block))
    return states
