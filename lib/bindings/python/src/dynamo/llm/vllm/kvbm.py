# SPDX-License-Identifier: Apache-2.0

from typing import List

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import KVCacheBlock

from dynamo._core import DynamoKvBlockList

logger = init_logger(__name__)


class KvbmCacheBlocks:
    """
    Implements the KVCacheBlocksProtocol interface.
    """

    def __init__(self, blocks: DynamoKvBlockList):
        self._owned_blocks = blocks
        self._blocks = [
            KVCacheBlock(
                block_id=blocks.get_block_id(i), _block_hash=blocks.get_block_hash(i)
            )
            for i in range(len(blocks))
        ]

    @property
    def blocks(self) -> List[KVCacheBlock]:
        return self._blocks

    def get_block_ids(self) -> list[list[int]]:
        return [[block.block_id for block in self.blocks]]

    def get_unhashed_block_ids(self) -> list[int]:
        return [block.block_id for block in self.blocks if block.block_hash is None]

    def __add__(self, other: "KvbmCacheBlocks") -> "KvbmCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        raise NotImplementedError("__add__ not implemented")

    @classmethod
    def create_empty(cls) -> "KvbmCacheBlocks":
        """Creates a new KVCacheBlocks instance with no blocks."""
        raise NotImplementedError("create_empty not implemented")
