# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A rank's KV events in vLLM's wire format (the ``EventSink`` ``RankRunner`` uses).

Dynamo's ``ZmqSource`` subscribes to it as to a vLLM worker: multipart
``[topic, 8-byte big-endian sequence, msgpack KVEventBatch]``, with
``data_parallel_rank`` set on every batch.
"""

from __future__ import annotations

import time

from .base import KVEvent
from .vllm.kv_events import (
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    ZmqEventPublisher,
)


def to_vllm_event(
    event: KVEvent, block_size: int
) -> BlockStored | BlockRemoved | AllBlocksCleared:
    if event.kind == "stored":
        return BlockStored(
            block_hashes=list(event.block_hashes),
            parent_block_hash=event.parent_block_hash,
            token_ids=list(event.token_ids),
            block_size=block_size,
            lora_id=None,
            medium=MEDIUM_GPU,
            lora_name=None,
        )
    if event.kind == "removed":
        return BlockRemoved(block_hashes=list(event.block_hashes), medium=MEDIUM_GPU)
    if event.kind == "cleared":
        return AllBlocksCleared()
    raise ValueError(f"unknown KV event kind {event.kind!r}")


class ZmqEventSink:
    """Publishes one ``KVEventBatch`` per call on ``endpoint`` (bound here)."""

    def __init__(self, endpoint: str, dp_rank: int, block_size: int = 128) -> None:
        # The endpoint already carries this rank's port. Rank 0 here keeps the
        # publisher from offsetting it again; every batch names dp_rank itself.
        self._publisher = ZmqEventPublisher(data_parallel_rank=0, endpoint=endpoint)
        self.dp_rank = dp_rank
        self.block_size = block_size

    def publish(self, events: list[KVEvent]) -> None:
        self._publisher.publish(
            KVEventBatch(
                ts=time.time(),
                events=[to_vllm_event(e, self.block_size) for e in events],
                data_parallel_rank=self.dp_rank,
            )
        )

    def close(self) -> None:
        self._publisher.shutdown()
