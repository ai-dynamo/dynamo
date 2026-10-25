# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable per-rank KV cache manager over DeepInfer's block allocator.

One manager per DP rank, owned by that rank's ``DynamoLLMEngine``. Every
method runs on the thread that owns the engine. Rank worker processes import
this module, so it must not import ``dynamo._core``, torch or DeepInfer.
"""

from __future__ import annotations

import importlib
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

import numpy as np

# DeepInfer's NORMAL scheduler prefills only while block usage is at most 90%
# (GPU_CACHE_PREFILL_WATERMARK, csrc/engine/scheduler.hpp:80,313), and a decode
# step preempts running seqs whenever no block is free (:628-640).
PREFILL_FREE_FRACTION = 0.1


def admission_reserve(engine: Any, fraction: float = PREFILL_FREE_FRACTION) -> int:
    """Free blocks an allocation made ahead of DeepInfer's scheduler (prefix
    hits, decode KV reads) must leave: the prefill headroom plus one block per
    running seq for decode growth."""
    total = int(engine.cache_config.num_gpu_blocks)
    return math.ceil(fraction * total) + len(engine.scheduler.running)


@dataclass
class Admission:
    """Result of :meth:`KVCacheManager.admit`."""

    # False means LATER: no room yet; the rank retries on a later iteration.
    admitted: bool
    # Prompt tokens served from cache (0 on a miss).
    hit_tokens: int = 0


@dataclass
class HeldKV:
    """KV pinned after prefill until a decode peer has pulled it."""

    request_id: str
    block_ids: list[int]
    state_id: int
    num_tokens: int
    first_token: int


@dataclass
class CacheStats:
    used_blocks: int
    total_blocks: int
    # None when there is no prefix cache (the gauge is skipped), else hits/queries.
    hit_rate: Optional[float] = None


@dataclass
class KVEvent:
    """A cache event in vLLM's model (``BlockStored``/``BlockRemoved``/``AllBlocksCleared``).

    ``block_hashes`` are the 64-bit vLLM-style chained hashes; for ``stored``
    events ``token_ids`` holds ``len(block_hashes) * block_size`` tokens and
    ``parent_block_hash`` is the hash of the block before the first one.
    """

    kind: Literal["stored", "removed", "cleared"]
    block_hashes: list[int] = field(default_factory=list)
    parent_block_hash: Optional[int] = None
    token_ids: list[int] = field(default_factory=list)
    block_size: int = 128


class KVCacheManager(ABC):
    """Admission, retention and eviction of KV for one DeepInfer engine.

    Constructed with the engine (a ``DynamoLLMEngine``) after DeepInfer built it;
    use ``engine.scheduler``, ``engine.block_manager``, ``engine.state_manager``,
    ``engine.cache_config`` and ``engine.driver_worker.cache_engine``.
    """

    block_size: int = 128

    def __init__(self, engine: Any, **options: Any) -> None:
        self.engine = engine
        self.block_size = int(engine.cache_config.block_size)

    @abstractmethod
    def admit(self, seq: Any, token_ids: np.ndarray) -> Admission:
        """Called after ``add_request(enqueue=False)``. Hand the seq to the
        scheduler, or return ``Admission(False)`` to be retried later.

        A miss must not allocate: DeepInfer allocates when it schedules the
        prefill, behind its 90% usage gate. A hit allocates with its shared
        prefix, restores state and sets ``num_cached_tokens``, and must leave
        :func:`admission_reserve` blocks free."""

    def before_step(self) -> None:
        """Called once per iteration before ``step()``: make room for the
        prompt DeepInfer's scheduler allocates next."""

    def on_post_schedule(self, scheduler_outputs: Any, engine: Any) -> None:
        """``LLMEngine.post_schedule`` hook: take snapshots at checkpoints."""

    @abstractmethod
    def on_finished(self, seq: Any) -> None:
        """``LLMEngine.finalize_seq`` hook: retain what is cacheable, free the rest."""

    def ensure_free(self, num_blocks: int) -> int:
        """Evict retained entries until ``num_blocks`` are free; returns blocks freed."""
        return 0

    def take_events(self) -> list[KVEvent]:
        """Drain the KV events produced since the last call."""
        return []

    @abstractmethod
    def stats(self) -> CacheStats:
        ...

    def close(self) -> None:
        """The engine is closing: finish background work that uses host memory."""

    @abstractmethod
    def hold_for_transfer(self, seq: Any) -> HeldKV:
        """Prefill role: pin the seq's blocks and state slot, release the rest of
        the seq (MTP slot, scheduler entry) and take it out of ``running``."""

    @abstractmethod
    def release_hold(self, held: HeldKV) -> None:
        """Unpin what :meth:`hold_for_transfer` pinned."""


class EventSink(Protocol):
    """Where a rank sends its :class:`KVEvent`s once per iteration."""

    def publish(self, events: list[KVEvent]) -> None:
        ...

    def close(self) -> None:
        ...


class NullEventSink:
    def publish(self, events: list[KVEvent]) -> None:
        pass

    def close(self) -> None:
        pass


# name -> (module relative to this package, class). Imported lazily so the
# default path never loads the prefix cache or its vendored dependencies.
_MANAGERS = {
    "none": (".noop", "NoopCacheManager"),
    "prefix": (".prefix", "PrefixCacheManager"),
}


def make_cache_manager(name: str, engine: Any, **options: Any) -> KVCacheManager:
    try:
        module_name, cls_name = _MANAGERS[name]
    except KeyError:
        raise ValueError(
            f"unknown kv cache manager {name!r}; expected one of {sorted(_MANAGERS)}"
        ) from None
    module = importlib.import_module(module_name, __package__)
    return getattr(module, cls_name)(engine, **options)


def make_event_sink(
    endpoint: Optional[str], dp_rank: int, block_size: int = 128
) -> EventSink:
    """Bind a vLLM-format ZMQ publisher on ``endpoint`` (``None``: no events).

    The publisher lives in ``kv_cache/events.py`` as ``ZmqEventSink(endpoint,
    dp_rank, block_size)``: it converts :class:`KVEvent` into the vendored
    vLLM ``KVEventBatch`` and publishes it with ``ZmqEventPublisher``.
    """
    if endpoint is None:
        return NullEventSink()
    module = importlib.import_module(".events", __package__)
    return module.ZmqEventSink(endpoint, dp_rank, block_size)
