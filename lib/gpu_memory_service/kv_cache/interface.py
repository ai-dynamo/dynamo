# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A block index that outlives the engine that built it.

GMS keeps the KV *bytes* across a failover; the bytes are unusable without the
index that finds them -- ``block_hash -> block_id``, which vLLM holds in process
memory and cannot rebuild, since the hash is over token ids rather than content.

Knows nothing of vLLM: block hashes are opaque bytes, compatibility is an opaque
digest the caller computes.

    BlockIndexStore  record / drop / drop_all / retain   write the index
                     usable()                            entries whose KV exists
                     on_schedule / on_complete           the batch counters
    Refusal          why a store was not admitted
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Iterator, Protocol, runtime_checkable


class Refusal(str, Enum):
    """Why a store was not admitted. Every value degrades to a cold cache."""

    OK = "ok"
    ABSENT = "absent"
    UNREADABLE = "unreadable"
    MAGIC = "magic"
    VERSION = "version"
    NUM_BLOCKS = "num_blocks"
    TRUNCATED = "truncated"
    IDENTITY = "identity"


def store_path() -> str | None:
    """Where stores live, or None when the feature is off (the default)."""
    return os.environ.get("GMS_KV_INDEX_PATH") or None


@runtime_checkable
class BlockIndexStore(Protocol):
    """Somewhere to keep a block index across a handover.

    An entry binds an opaque ``block_hash`` to a ``block_id``. The binding is
    only true while the block still holds those bytes, so the store also carries
    two batch counters: an entry recorded during batch ``scheduled`` becomes
    usable once ``completed`` reaches it. vLLM names blocks while *planning* a
    batch, before the KV exists, and an engine that dies mid-batch never
    advances ``completed`` that far.
    """

    num_blocks: int
    scheduled: int
    completed: int

    # -- the batch counters -------------------------------------------------
    def on_schedule(self) -> None:
        """A batch has started."""

    def on_complete(self) -> None:
        """A batch has finished; its entries are now usable."""

    # -- writing ------------------------------------------------------------
    def record(self, block_id: int, block_hash: bytes, num_tokens: int | None) -> None:
        """Bind ``block_hash`` to ``block_id`` as of the current batch."""

    def drop(self, block_id: int) -> None:
        """Forget one entry."""

    def drop_all(self) -> None:
        """Forget every entry."""

    def retain(self, block_ids) -> None:
        """Forget every entry except these."""

    # -- reading ------------------------------------------------------------
    def usable(self) -> Iterator[tuple[int, bytes, int | None]]:
        """Entries whose KV is known to exist: ``(block_id, hash, num_tokens)``."""

    def live(self) -> set[int]:
        """Every claimed block, ignoring the counters. Tests assert on this."""

    def close(self) -> None:
        ...
