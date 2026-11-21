# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-process KV transfer for tests and the CPU fake.

Instances find each other through a process-wide registry keyed by agent name.
A read copies the peer's rows when it completes (at a later ``poll``), and,
like a NIXL READ, a completed read notifies the peer that it may free the KV.
"""

from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from .base import KVLayout, KVRegion, KVTransfer
from .rows import ReadPlan, plan_read

if TYPE_CHECKING:
    from dynamo.deepinfer.disagg import DeepInferHandoff

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, LocalKVTransfer] = {}
_REGISTRY_LOCK = threading.Lock()


def _peer(agent_name: str) -> Optional[LocalKVTransfer]:
    with _REGISTRY_LOCK:
        return _REGISTRY.get(agent_name)


@dataclass
class _Read:
    handoff: DeepInferHandoff
    plan: ReadPlan
    polls_left: int


class LocalKVTransfer(KVTransfer):
    """``delay_polls``: polls a read stays in flight after the first (tests use it
    to abort mid-transfer)."""

    def __init__(
        self, *, rank: int, engine_id: str, delay_polls: int = 0, **options: Any
    ) -> None:
        super().__init__(rank=rank, engine_id=engine_id)
        self.agent_name = f"{engine_id}-dp{rank}-{uuid.uuid4().hex[:8]}"
        self.delay_polls = delay_polls
        self._layout: Optional[KVLayout] = None
        self._reads: dict[str, _Read] = {}
        # Prefill-side request ids whose KV a reader finished with; peers
        # append from their own threads.
        self._released: list[str] = []
        self._released_lock = threading.Lock()
        # (agent, request) pairs a completed read already released.
        self._notified: set[tuple[str, str]] = set()

    def register(self, layout: KVLayout) -> None:
        if any(r.buffer is None for r in layout.regions):
            raise ValueError("LocalKVTransfer needs each region's buffer")
        self._layout = layout
        with _REGISTRY_LOCK:
            _REGISTRY[self.agent_name] = self

    def agent_info(self) -> dict[str, str]:
        return {"agent_name": self.agent_name, "agent_metadata": ""}

    def start_read(
        self,
        req_id: str,
        handoff: DeepInferHandoff,
        local_block_ids: list[int],
        local_state_id: int,
    ) -> None:
        if self._layout is None:
            raise RuntimeError("register() must run before start_read()")
        if req_id in self._reads:
            raise ValueError(f"a read for {req_id} is already in flight")
        peer = _peer(handoff.agent_name)
        if peer is None or peer._layout is None:
            raise ConnectionError(f"unknown KV peer {handoff.agent_name!r}")
        shapes = {r.name: (r.num_items, r.item_bytes) for r in peer._layout.regions}
        plan = plan_read(self._layout, shapes, handoff, local_block_ids, local_state_id)
        self._reads[req_id] = _Read(handoff, plan, 1 + self.delay_polls)

    def poll(self) -> tuple[list[str], list[str], list[str]]:
        done: list[str] = []
        failed: list[str] = []
        for rid, read in list(self._reads.items()):
            read.polls_left -= 1
            if read.polls_left > 0:
                continue
            del self._reads[rid]
            peer = _peer(read.handoff.agent_name)
            if peer is None or peer._layout is None:
                logger.error(
                    "KV peer %s went away during a read", read.handoff.agent_name
                )
                failed.append(rid)
                continue
            assert self._layout is not None
            for name, dst, src in read.plan:
                _copy_rows(
                    self._layout.region(name), dst, peer._layout.region(name), src
                )
            peer._notify(read.handoff.request_id)
            self._notified.add((read.handoff.agent_name, read.handoff.request_id))
            done.append(rid)
        with self._released_lock:
            released, self._released = self._released, []
        return done, failed, released

    def release_remote(self, handoff: DeepInferHandoff) -> None:
        key = (handoff.agent_name, handoff.request_id)
        if key in self._notified:
            self._notified.discard(key)
            return
        peer = _peer(handoff.agent_name)
        if peer is not None:
            peer._notify(handoff.request_id)

    def shutdown(self) -> None:
        with _REGISTRY_LOCK:
            if _REGISTRY.get(self.agent_name) is self:
                del _REGISTRY[self.agent_name]
        self._reads.clear()

    def _notify(self, request_id: str) -> None:
        with self._released_lock:
            self._released.append(request_id)


def _copy_rows(
    dst: KVRegion, dst_ids: list[int], src: KVRegion, src_ids: list[int]
) -> None:
    rows = src.buffer[src_ids]
    if hasattr(rows, "to"):  # torch: the peer may sit on another GPU
        rows = rows.to(dst.buffer.device)
    dst.buffer[dst_ids] = rows
