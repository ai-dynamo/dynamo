# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV-event source descriptors returned by :meth:`LLMEngine.kv_event_sources`
plus the :class:`ComponentSnapshot` payload engines push into the
framework-owned :class:`SnapshotPublisher` (declared via
:meth:`LLMEngine.component_metrics_dp_ranks` + received via
:meth:`LLMEngine.attach_snapshot_publisher`). Worker constructs the
publisher; engines never instantiate one themselves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Union

if TYPE_CHECKING:
    from dynamo.llm import KvEventPublisher


@dataclass(frozen=True)
class ZmqSource:
    """Worker subscribes to the engine's ZMQ PUB socket directly."""

    endpoint: str
    topic: str = ""
    dp_rank: int = 0
    image_token_id: Optional[int] = None


@dataclass(frozen=True)
class PushSource:
    """Worker hands a live ``KvEventPublisher`` to the engine via
    ``on_ready``; the engine drives ``publish`` from its own thread and
    MUST stop that thread in :meth:`LLMEngine.cleanup` before returning."""

    on_ready: Callable[["KvEventPublisher"], None]
    dp_rank: int = 0


@dataclass
class ComponentSnapshot:
    """Rich per-rank snapshot the engine pushes into the Rust-owned
    :class:`SnapshotPublisher` from its stat-logger thread.

    ``SnapshotPublisher.publish(rank, snap)`` atomically updates the
    per-rank ``dynamo_component_*`` gauges (``total_blocks``,
    ``gpu_cache_usage_percent``, ``kv_cache_hit_rate``) AND fires the
    router-input signal (``kv_used_blocks``) inline — no GIL on the
    framework reader side, no polling.

    ``kv_cache_hit_rate`` is tri-state:
    - ``None``: engine hasn't observed requests yet OR has no prefix
      cache (gauge skipped — distinguishes "no measurement" from "0%").
    - ``0.0``: legitimate measurement (no hits).
    """

    kv_used_blocks: int
    kv_total_blocks: int
    gpu_cache_usage: float
    dp_rank: int
    kv_cache_hit_rate: Optional[float] = None


KvEventSource = Union[ZmqSource, PushSource]


__all__ = [
    "ComponentSnapshot",
    "KvEventSource",
    "PushSource",
    "ZmqSource",
]
