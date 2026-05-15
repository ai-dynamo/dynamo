# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source descriptors returned by :meth:`LLMEngine.kv_event_sources` and
:meth:`LLMEngine.metrics_sources`. Worker constructs one publisher per
descriptor — engines never instantiate publishers themselves."""

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


@dataclass(frozen=True)
class SnapshotSource:
    """``snapshot`` is invoked under the GIL on a fixed interval. Keep it to a
    member-field read; return ``None`` to skip publishing for that tick."""

    snapshot: Callable[[], Optional["Metrics"]]
    dp_rank: int = 0


KvEventSource = Union[ZmqSource, PushSource]


@dataclass
class Metrics:
    """Router input. Lower ``kv_used_blocks`` = more free capacity."""

    kv_used_blocks: Optional[int] = None


__all__ = [
    "KvEventSource",
    "Metrics",
    "PushSource",
    "SnapshotSource",
    "ZmqSource",
]
