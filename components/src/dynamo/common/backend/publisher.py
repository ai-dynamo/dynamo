# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source descriptors returned by :meth:`LLMEngine.kv_event_sources` and
:meth:`LLMEngine.component_metrics_sources`. Worker constructs one publisher
per descriptor — engines never instantiate publishers themselves."""

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


@dataclass(frozen=True)
class PushSource:
    """Worker hands a live ``KvEventPublisher`` to the engine via
    ``on_ready``; the engine drives ``publish`` from its own thread and
    MUST stop that thread in :meth:`LLMEngine.cleanup` before returning."""

    on_ready: Callable[["KvEventPublisher"], None]
    dp_rank: int = 0


@dataclass
class ComponentSnapshot:
    """Rich per-rank snapshot returned by
    :class:`ComponentMetricsSource`. ``Worker`` consumes it to drive both
    the router-input signal (``kv_used_blocks``) and the engine-side
    ``dynamo_component_*`` gauges (``kv_total_blocks``, ``gpu_cache_usage``).

    Engines fill an in-memory dict from their natural push surface
    (stat-logger / ZMQ / poll thread); the snapshot fn just returns the
    latest entry as a cheap field read.
    """

    kv_used_blocks: int
    kv_total_blocks: int
    gpu_cache_usage: float
    dp_rank: int


@dataclass(frozen=True)
class ComponentMetricsSource:
    """``snapshot`` is invoked under the GIL on a fixed interval. Keep it to
    a member-field read; return ``None`` to skip publishing for that tick.
    Conformance kit enforces a 1 ms ceiling."""

    snapshot: Callable[[], Optional[ComponentSnapshot]]
    dp_rank: int = 0


KvEventSource = Union[ZmqSource, PushSource]


__all__ = [
    "ComponentMetricsSource",
    "ComponentSnapshot",
    "KvEventSource",
    "PushSource",
    "ZmqSource",
]
